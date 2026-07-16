"""Signal-occurrence diagnostics (Layer 0: measurement only, no behavior change).

The backtest's attribution table counts signals attached to EXECUTED entries.
That conflates three different questions:

  1. "The signal occurred."           -> raw occurrence count
  2. "The signal qualified as entry." -> passed primary + confidence gates while flat
  3. "The order was executed."        -> risk manager approved and a position opened

This module answers all three independently, and produces a per-event trace for
bullish Kumo twists explaining exactly why each one did or did not become a trade.

It replays the SAME executor/risk pipeline as ichibot.backtest, but computes
signal flags once, vectorially (all rolling operations in signals_per_row are
causal, so per-row flags are identical to the incremental window computation
used by replay_history -- verified in tests). This makes the replay O(n).

Run:  python -m ichibot.diagnostics --days 2000
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ichibot.executor_dryrun import DryRunExecutor
from ichibot.ichimoku import compute_ichimoku, min_required_candles
from ichibot.risk import RiskManager
from ichibot.signals import (
    ALL_SIGNALS,
    PRIMARY_BULLISH,
    SIG_KUMO_TWIST_BULL,
    SignalWeights,
    result_from_flags,
    signals_per_row,
)


@dataclass
class SignalCounts:
    """Occurrence counters for one signal in one market."""
    raw: int = 0                 # every row where the flag is True
    after_warmup: int = 0        # ... with a fully-formed cloud (tradeable region)
    while_flat: int = 0          # ... and no open position in this market
    while_in_position: int = 0   # ... with a position already open
    met_confidence: int = 0      # ... flat, and total bar confidence >= threshold
    with_primary: int = 0        # ... flat, co-firing with >= 1 primary signal
    executed_entry: int = 0      # ... on the candle of an actually opened trade


@dataclass
class TwistEvent:
    """One bullish Kumo twist and the precise reason it did / didn't trade."""
    timestamp: str
    price_vs_cloud: str          # ABOVE_CLOUD / IN_CLOUD / BELOW_CLOUD / UNKNOWN
    tenkan_above_kijun: bool
    coincident_bullish: tuple
    confidence: float
    position_open: bool
    verdict: str


@dataclass
class MarketDiagnostics:
    coin: str
    counts: dict = field(default_factory=dict)          # signal -> SignalCounts
    twist_events: list = field(default_factory=list)    # list[TwistEvent]
    warmup_bars: int = 0
    total_bars: int = 0


def _cloud_position(row: pd.Series) -> str:
    top, bottom = row["cloud_top"], row["cloud_bottom"]
    if pd.isna(top) or pd.isna(bottom):
        return "UNKNOWN"
    if row["close"] > top:
        return "ABOVE_CLOUD"
    if row["close"] < bottom:
        return "BELOW_CLOUD"
    return "IN_CLOUD"


def _twist_verdict(*, position_open: bool, primaries: list, confidence: float,
                   min_confidence: float, opened: bool) -> str:
    if opened:
        return "coincided with an executed entry (attribution credits the twist)"
    if position_open:
        return "position already open in this market"
    reasons = []
    if not primaries:
        reasons.append("not a primary signal and no primary co-fired")
    if confidence < min_confidence:
        reasons.append(f"confidence {confidence:.2f} < threshold {min_confidence:.2f}")
    if not reasons:  # gates passed but risk manager rejected (e.g. exposure)
        reasons.append("entry gates passed but sizing/exposure rejected the order")
    return "; ".join(reasons)


def diagnose_market(
    coin: str,
    ich: pd.DataFrame,
    risk: RiskManager,
    min_confidence: float,
    logger: logging.Logger,
    weights: SignalWeights | None = None,
) -> MarketDiagnostics:
    """Replay one market through the live executor pipeline, counting every
    signal occurrence independently of trade execution."""
    weights = weights or SignalWeights()
    flags = signals_per_row(ich)  # computed ONCE; rolling ops are causal
    warm = ich["cloud_top"].notna()

    diag = MarketDiagnostics(coin=coin, total_bars=len(ich),
                             warmup_bars=int((~warm).sum()),
                             counts={s: SignalCounts() for s in ALL_SIGNALS})
    ex = DryRunExecutor(risk, logger, store=None)

    for i in range(1, len(ich)):
        row = ich.iloc[i]
        f = flags.iloc[i]
        sig = result_from_flags(f, row.get("time"), min_confidence, weights)
        bullish, bearish, confidence = sig.bullish_signals, sig.bearish_signals, sig.confidence
        primaries = [s for s in bullish if s in PRIMARY_BULLISH]
        flat_before = coin not in ex.positions

        action = ex.process(coin, float(row["close"]), sig)
        opened = action == "opened"

        for s in bullish + bearish:
            c = diag.counts[s]
            c.raw += 1
            if not warm.iloc[i]:
                continue
            c.after_warmup += 1
            if flat_before:
                c.while_flat += 1
                if confidence >= min_confidence:
                    c.met_confidence += 1
                if primaries:
                    c.with_primary += 1
                if opened:
                    c.executed_entry += 1
            else:
                c.while_in_position += 1

        if f[SIG_KUMO_TWIST_BULL] and warm.iloc[i]:
            ts = str(row["time"].date()) if "time" in ich.columns else str(i)
            diag.twist_events.append(TwistEvent(
                timestamp=ts,
                price_vs_cloud=_cloud_position(row),
                tenkan_above_kijun=bool(row["tenkan"] > row["kijun"]),
                coincident_bullish=tuple(s for s in bullish if s != SIG_KUMO_TWIST_BULL),
                confidence=confidence,
                position_open=not flat_before,
                verdict=_twist_verdict(position_open=not flat_before,
                                       primaries=primaries, confidence=confidence,
                                       min_confidence=min_confidence, opened=opened),
            ))
    return diag


def format_report(diags: list[MarketDiagnostics]) -> str:
    lines = ["Signal-occurrence diagnostics",
             "'raw' counts every firing; 'entry' counts candles where a trade actually opened.", ""]
    header = (f"{'MARKET':<7}{'SIGNAL':<28}{'RAW':>6}{'WARM':>6}{'FLAT':>6}"
              f"{'INPOS':>6}{'CONF>=':>7}{'W/PRIM':>7}{'ENTRY':>6}")
    lines += [header, "-" * len(header)]
    for d in diags:
        for s in ALL_SIGNALS:
            c = d.counts[s]
            if c.raw == 0:
                continue
            lines.append(f"{d.coin:<7}{s:<28}{c.raw:>6}{c.after_warmup:>6}"
                         f"{c.while_flat:>6}{c.while_in_position:>6}"
                         f"{c.met_confidence:>7}{c.with_primary:>7}{c.executed_entry:>6}")
        lines.append("")
    lines.append("Bullish Kumo twist trace (why each twist did or did not trade):")
    lines.append("")
    for d in diags:
        for e in d.twist_events:
            co = ",".join(e.coincident_bullish) or "-"
            lines.append(f"  {d.coin:<6}{e.timestamp}  {e.price_vs_cloud:<12}"
                         f"TK{'>' if e.tenkan_above_kijun else '<='}KJ  "
                         f"conf={e.confidence:.2f}  co=[{co}]  -> {e.verdict}")
        if not d.twist_events:
            lines.append(f"  {d.coin:<6}(no bullish twists after warm-up)")
    return "\n".join(lines)


def main() -> int:
    import argparse
    from ichibot.config import load_config, ConfigError
    from ichibot.logging_setup import setup_logging
    from ichibot.market_data import HyperliquidData, MarketDataError

    parser = argparse.ArgumentParser(description="ichibot signal diagnostics")
    parser.add_argument("--days", type=int, default=2000)
    args = parser.parse_args()

    log = setup_logging(); log.setLevel(logging.WARNING)
    try:
        cfg = load_config("config.yaml", ".env")
    except ConfigError as exc:
        print(f"Configuration error: {exc}"); return 1
    try:
        data = HyperliquidData()
    except MarketDataError as exc:
        print(f"Market data unavailable: {exc}"); return 1

    needed = min_required_candles(cfg.ichimoku.span_b_periods, cfg.ichimoku.displacement)
    diags = []
    for coin in cfg.trading.markets:
        df = data.fetch_daily(coin, lookback_days=args.days,
                              drop_incomplete=cfg.trading.only_completed_candles)
        if len(df) < needed:
            print(f"{coin}: only {len(df)} candles (< {needed} warm-up); skipping")
            continue
        ich = compute_ichimoku(df, conversion_periods=cfg.ichimoku.conversion_periods,
                               base_periods=cfg.ichimoku.base_periods,
                               span_b_periods=cfg.ichimoku.span_b_periods,
                               displacement=cfg.ichimoku.displacement)
        risk = RiskManager.from_config(cfg.risk, cfg.trading.max_leverage)
        diags.append(diagnose_market(coin, ich, risk,
                                     cfg.risk.min_signal_confidence, log))
    print(format_report(diags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())