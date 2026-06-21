"""Dry-run (paper) order executor (Milestone 6).

Takes risk-checked signal decisions and simulates execution: opens/closes paper
positions, runs SL / TP / trailing-stop / signal exits each run,
and persists open positions to data/positions.json so a paper long opened one day
is still tracked the next.

This module never places a real order. It logs simulated fills only. Real order
placement lives in the separate executor_live.py module.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from ichibot.risk import Position, RiskManager
from ichibot.signals import SignalResult


class ExecutorError(Exception):
    """Raised on unrecoverable executor/persistence problems."""


class PositionStore:
    """Loads/saves paper positions to a JSON file, with atomic writes."""

    def __init__(self, path: str = "data/positions.json"):
        self.path = Path(path)

    def load(self) -> dict[str, Position]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ExecutorError(
                f"Could not read positions file {self.path}: {exc}. "
                f"Fix or delete it to start fresh."
            ) from exc
        try:
            return {coin: Position(**pdata) for coin, pdata in raw.items()}
        except TypeError as exc:
            raise ExecutorError(
                f"Positions file {self.path} has an unexpected shape: {exc}"
            ) from exc

    def save(self, positions: dict[str, Position]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {coin: asdict(pos) for coin, pos in positions.items()}
        text = json.dumps(data, indent=2, sort_keys=True)
        # Atomic write: write a temp file, then replace, so an interrupted run
        # can never leave a half-written positions file.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)


class DryRunExecutor:
    """Simulates trade execution against in-memory paper positions."""

    def __init__(self, risk_manager: RiskManager, logger, store: PositionStore | None = None):
        self.risk = risk_manager
        self.log = logger
        self.store = store
        self.positions: dict[str, Position] = store.load() if store else {}
        self.realized_pnl: float = 0.0

    def current_exposure_usd(self) -> float:
        return sum(p.notional_usd for p in self.positions.values())

    def process(self, coin: str, price: float, signal: SignalResult) -> str:
        """Process one market for one completed candle. Returns an action string:
        opened / hold / closed:<reason> / entry_rejected / none."""
        if coin in self.positions:
            return self._manage_open(coin, price, signal)
        return self._consider_entry(coin, price, signal)

    def _manage_open(self, coin: str, price: float, signal: SignalResult) -> str:
        pos = self.positions[coin]
        self.risk.update_trailing_peak(pos, price)

        exit_dec = self.risk.evaluate_exit(pos, price)
        if exit_dec.should_exit:
            return self.close_position(coin, price, exit_dec.reason)

        if signal.exit_recommended:
            return self.close_position(coin, price, "signal:" + ",".join(signal.bearish_signals))

        self.log.info("[DRY-RUN] HOLD %-6s @ %.4f (entry %.4f, stop %.4f)",
                      coin, price, pos.entry_price, pos.stop_price)
        return "hold"

    def _consider_entry(self, coin: str, price: float, signal: SignalResult) -> str:
        if not signal.entry_recommended:
            return "none"
        decision = self.risk.size_position(
            coin, price, signal.confidence, current_exposure_usd=self.current_exposure_usd()
        )
        if not decision.approved:
            self.log.info("[DRY-RUN] ENTRY SKIPPED %-6s: %s", coin, decision.reason)
            return "entry_rejected"
        pos = Position.from_decision(decision)
        self.positions[coin] = pos
        tp = f"{pos.take_profit_price:.4f}" if pos.take_profit_price is not None else "off"
        self.log.info(
            "[DRY-RUN] OPEN LONG %-6s @ %.4f | %.6f units ($%.2f) | stop %.4f tp %s | conf %.2f [%s]",
            coin, price, pos.size_units, pos.notional_usd, pos.stop_price, tp,
            signal.confidence, ",".join(signal.bullish_signals),
        )
        return "opened"

    def close_position(self, coin: str, price: float, reason: str = "manual") -> str:
        """Close an open paper position at `price`, recording realized PnL."""
        if coin not in self.positions:
            return "none"
        pos = self.positions.pop(coin)
        pnl = (price - pos.entry_price) * pos.size_units
        pnl_pct = (price / pos.entry_price - 1.0) * 100.0
        self.realized_pnl += pnl
        self.log.info(
            "[DRY-RUN] CLOSE LONG %-6s @ %.4f | entry %.4f | pnl $%.2f (%.2f%%) | reason %s",
            coin, price, pos.entry_price, pnl, pnl_pct, reason,
        )
        return f"closed:{reason}"

    def commit(self) -> None:
        """Persist current paper positions (no-op if constructed without a store)."""
        if self.store:
            self.store.save(self.positions)