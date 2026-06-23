"""Orchestration engine + scheduler (Milestone 7).

`Engine.run_once()` performs one daily pass over the configured markets:
fetch completed candles -> Ichimoku -> signals -> dry-run executor. 
Duplicate trade-guard: Idempotent, records the last completed-candle date it acted on for each market
in data/engine_state.json, so running it more than once for the same daily candle
does nothing the second time. 

`run_forever()` runs once now, then once shortly after each 00:00 UTC. For
production, an OS scheduler (cron/launchd) invoking `python main.py` is usually
more robust than keeping a long-lived process alive.

This module does not import the exchange client or the executor concretely -- they
are injected -- so it stays easy to test with fakes.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ichibot.ichimoku import cloud_position, compute_ichimoku, min_required_candles
from ichibot.signals import evaluate_signals


class RunStateStore:
    """Persists the last completed-candle date processed per market."""

    def __init__(self, path: str = "data/engine_state.json", logger=None):
        self.path = Path(path)
        self.log = logger

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            if self.log:
                self.log.warning("Run-state file unreadable (%s); starting fresh.", exc)
            return {}

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)


def seconds_until_next_utc_run(buffer_minutes: int = 5, now: datetime | None = None) -> float:
    """Seconds from `now` until just after the next 00:00 UTC (+buffer_minutes)."""
    now = now or datetime.now(timezone.utc)
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return ((next_midnight + timedelta(minutes=buffer_minutes)) - now).total_seconds()


class Engine:
    """Ties data -> Ichimoku -> signals -> executor together for one daily pass."""

    def __init__(self, cfg, data, executor, logger, state_path: str = "data/engine_state.json"):
        self.cfg = cfg
        self.data = data
        self.executor = executor
        self.log = logger
        self.state_store = RunStateStore(state_path, logger)
        self.needed = min_required_candles(cfg.ichimoku.span_b_periods, cfg.ichimoku.displacement)

    def run_once(self) -> dict:
        state = self.state_store.load()
        last_processed = state.get("last_processed", {})
        summary = {"processed": [], "skipped_dup": [], "skipped_other": [], "actions": {}}

        self.log.info("Engine run: scanning %d market(s); need >= %d candles.",
                      len(self.cfg.trading.markets), self.needed)

        for coin in self.cfg.trading.markets:
            try:
                df = self.data.fetch_daily(
                    coin, lookback_days=200,
                    drop_incomplete=self.cfg.trading.only_completed_candles,
                )
            except Exception as exc:  # per-market isolation: one failure must not kill the run
                self.log.warning("Skipping %s: fetch failed: %s", coin, exc)
                summary["skipped_other"].append(coin)
                continue

            if len(df) < self.needed:
                self.log.warning("%s: only %d candles, need %d -- skipping.",
                                 coin, len(df), self.needed)
                summary["skipped_other"].append(coin)
                continue

            ich = compute_ichimoku(
                df,
                conversion_periods=self.cfg.ichimoku.conversion_periods,
                base_periods=self.cfg.ichimoku.base_periods,
                span_b_periods=self.cfg.ichimoku.span_b_periods,
                displacement=self.cfg.ichimoku.displacement,
            )
            last = ich.iloc[-1]
            candle_date = str(last["time"].date())

            if last_processed.get(coin) == candle_date:
                self.log.info("%-6s | candle %s already processed -- skipping.", coin, candle_date)
                summary["skipped_dup"].append(coin)
                continue

            price = float(last["close"])
            signal = evaluate_signals(ich, self.cfg.risk.min_signal_confidence)
            held = "HELD" if coin in getattr(self.executor, "positions", {}) else "flat"
            self.log.info("%-6s | %s close=%.4f | %s | %s | %s",
                          coin, candle_date, price, cloud_position(last), held, signal.summary())

            action = self.executor.process(coin, price, signal)
            last_processed[coin] = candle_date
            summary["processed"].append(coin)
            summary["actions"][coin] = action

        self.executor.commit()
        state["last_processed"] = last_processed
        self.state_store.save(state)

        self.log.info("Engine run done: %d processed, %d already-done, %d skipped.",
                      len(summary["processed"]), len(summary["skipped_dup"]),
                      len(summary["skipped_other"]))
        return summary

    def run_forever(self, buffer_minutes: int = 5) -> None:
        """Run once now, then once shortly after each 00:00 UTC. Ctrl-C to stop."""
        self.log.info("Scheduler loop started. Ctrl-C to stop.")
        while True:
            self.run_once()
            secs = seconds_until_next_utc_run(buffer_minutes)
            self.log.info("Next run in %.1f h (just after 00:00 UTC).", secs / 3600)
            time.sleep(secs)