"""Signal detection for the Ichimoku strategy (Milestone 4a: five core signals).

Stateless: evaluate_signals() looks only at the most recent completed candle and
the one before it (to detect crossings) in an Ichimoku DataFrame produced by
ichibot.ichimoku.compute_ichimoku. evaluate_signals() is unaware of open positions, order
history, or duplicate-trade suppression -- those belong to the engine (later).

Design choices worth noting: 
  * Entries are GATED: an entry is recommended only when bullish confidence meets/exceeds
    the configured threshold and at least one PRIMARY trigger fired. This stops a
    pile of weak confirmations from opening a trade on their own. 
  * Exits are not gated by confidence. Any single bearish exit signal recommends
    an exit. Waiting for "confident" exits would mean holding losers which is the wrong
    bias for capital preservation.
  * All five signals are crossing events (a transition between two candles), not
    persistent states, matching the spec's cross / breakout / breaking-below wording.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# --- Signal name constants (used as dict keys and log labels) --------------
SIG_TK_CROSS_BULL = "bullish_tk_cross"
SIG_CLOUD_BREAKOUT = "price_breakout_above_cloud"
SIG_KUMO_TWIST_BULL = "bullish_kumo_twist"
SIG_BELOW_TENKAN = "price_below_tenkan"
SIG_BELOW_KIJUN = "price_below_kijun"

BULLISH_SIGNALS = (SIG_CLOUD_BREAKOUT, SIG_TK_CROSS_BULL, SIG_KUMO_TWIST_BULL)
BEARISH_SIGNALS = (SIG_BELOW_TENKAN, SIG_BELOW_KIJUN)
ALL_SIGNALS = BULLISH_SIGNALS + BEARISH_SIGNALS

# Primary bullish triggers: at least one must fire for an entry to be recommended.
PRIMARY_BULLISH = frozenset({SIG_CLOUD_BREAKOUT, SIG_TK_CROSS_BULL})

# Columns evaluate_signals needs from the Ichimoku DataFrame.
REQUIRED_COLUMNS = (
    "close", "tenkan", "kijun", "cloud_top",
    "senkou_a_future", "senkou_b_future",
)


@dataclass
class SignalWeights:
    """Starter weights for the bullish confidence score. TUNE THESE.

    They are starting points, not received wisdom. With min_signal_confidence=0.6:
      * a cloud breakout alone (0.6) just meets the bar,
      * a T/K cross alone (0.5) does not, but T/K + twist (0.75) does,
      * any two primaries hit the 1.0 cap.
    """
    cloud_breakout: float = 0.6
    tk_cross: float = 0.5
    kumo_twist: float = 0.25

    def weight_for(self, name: str) -> float:
        return {
            SIG_CLOUD_BREAKOUT: self.cloud_breakout,
            SIG_TK_CROSS_BULL: self.tk_cross,
            SIG_KUMO_TWIST_BULL: self.kumo_twist,
        }.get(name, 0.0)


@dataclass
class SignalResult:
    timestamp: object                 # candle open time of the evaluated bar
    bullish_signals: list[str]        # names of fired bullish signals
    bearish_signals: list[str]        # names of fired bearish exit signals
    confidence: float                 # 0..1 bullish confidence
    entry_recommended: bool
    exit_recommended: bool
    details: dict                     # per-signal {"fired": bool, "weight": float}

    def summary(self) -> str:
        bull = ",".join(self.bullish_signals) or "-"
        bear = ",".join(self.bearish_signals) or "-"
        return (f"conf={self.confidence:.2f} entry={self.entry_recommended} "
                f"exit={self.exit_recommended} bull=[{bull}] bear=[{bear}]")


def signals_per_row(ich: pd.DataFrame) -> pd.DataFrame:
    """Boolean DataFrame: for every row, which of the five signals fired vs the
    previous row. Vectorized; useful for history scans and (later) backtests.

    Comparisons against NaN evaluate to False, so warmup rows never fire.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in ich.columns]
    if missing:
        raise ValueError(f"Ichimoku DataFrame missing columns: {missing}")

    def cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.shift(1) <= b.shift(1)) & (a > b)

    def cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.shift(1) >= b.shift(1)) & (a < b)

    out = pd.DataFrame(index=ich.index)
    out[SIG_TK_CROSS_BULL] = cross_above(ich["tenkan"], ich["kijun"])
    out[SIG_CLOUD_BREAKOUT] = cross_above(ich["close"], ich["cloud_top"])
    out[SIG_KUMO_TWIST_BULL] = cross_above(ich["senkou_a_future"], ich["senkou_b_future"])
    out[SIG_BELOW_TENKAN] = cross_below(ich["close"], ich["tenkan"])
    out[SIG_BELOW_KIJUN] = cross_below(ich["close"], ich["kijun"])
    return out.astype(bool)


def _neutral_result(timestamp) -> SignalResult:
    return SignalResult(
        timestamp=timestamp,
        bullish_signals=[],
        bearish_signals=[],
        confidence=0.0,
        entry_recommended=False,
        exit_recommended=False,
        details={},
    )


def evaluate_signals(
    ich: pd.DataFrame,
    min_confidence: float,
    weights: SignalWeights | None = None,
) -> SignalResult:
    """Evaluate the five core signals on the latest completed candle.

    `ich` is an Ichimoku DataFrame (output of compute_ichimoku), sorted ascending
    by time. Returns a SignalResult describing what fired on the last row (using
    the previous row to detect crossings).
    """
    if weights is None:
        weights = SignalWeights()

    has_time = "time" in ich.columns
    if len(ich) < 2:
        ts = ich.iloc[-1]["time"] if (len(ich) == 1 and has_time) else None
        return _neutral_result(ts)

    flags = signals_per_row(ich)
    last = flags.iloc[-1]
    fired = {name: bool(last[name]) for name in ALL_SIGNALS}

    bullish = [name for name in BULLISH_SIGNALS if fired[name]]
    bearish = [name for name in BEARISH_SIGNALS if fired[name]]

    confidence = min(1.0, sum(weights.weight_for(name) for name in bullish))
    has_primary = any(name in PRIMARY_BULLISH for name in bullish)
    entry_recommended = confidence >= min_confidence and has_primary
    exit_recommended = len(bearish) > 0

    details = {
        name: {
            "fired": fired[name],
            "weight": weights.weight_for(name) if name in BULLISH_SIGNALS else 0.0,
        }
        for name in ALL_SIGNALS
    }

    ts = ich.iloc[-1]["time"] if has_time else None
    return SignalResult(
        timestamp=ts,
        bullish_signals=bullish,
        bearish_signals=bearish,
        confidence=confidence,
        entry_recommended=entry_recommended,
        exit_recommended=exit_recommended,
        details=details,
    )