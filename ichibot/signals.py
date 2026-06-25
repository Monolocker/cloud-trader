"""Signal detection for the Ichimoku strategy.

Milestone 4a: five core crossing signals.
Milestone 4b: three pattern signals (edge-to-edge, C-clamp, flat-Kijun), each
with a bullish entry variant and a bearish exit mirror.

Stateless: evaluate_signals() looks at the latest completed candle (and the bars
before it, for multi-bar patterns) in an Ichimoku DataFrame from compute_ichimoku.

Design notes:
  * Entries are gated: confidence must meet the threshold and at least one primary
    bullish trigger must fire. Exits are not gated -- any one bearish signal exits.
  * Pattern thresholds live in PatternParams (tunable in one place).
  * The bullish pattern variants are entries; the bearish mirrors are exits
    (long-only: bearish can only close a long).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# --- Core signal names (4a) ------------------------------------------------
SIG_TK_CROSS_BULL = "bullish_tk_cross"
SIG_CLOUD_BREAKOUT = "price_breakout_above_cloud"
SIG_KUMO_TWIST_BULL = "bullish_kumo_twist"
SIG_BELOW_TENKAN = "price_below_tenkan"
SIG_BELOW_KIJUN = "price_below_kijun"

# --- Pattern signal names (4b) ---------------------------------------------
SIG_E2E_BULL = "edge_to_edge_bull"
SIG_CCLAMP_BULL = "c_clamp_bull"
SIG_FLAT_KIJUN_BULL = "flat_kijun_bull"
SIG_E2E_BEAR = "edge_to_edge_bear"
SIG_CCLAMP_BEAR = "c_clamp_bear"
SIG_FLAT_KIJUN_BEAR = "flat_kijun_bear"

BULLISH_SIGNALS = (
    SIG_CLOUD_BREAKOUT, SIG_TK_CROSS_BULL, SIG_KUMO_TWIST_BULL,
    SIG_FLAT_KIJUN_BULL, SIG_CCLAMP_BULL, SIG_E2E_BULL,
)
BEARISH_SIGNALS = (
    SIG_BELOW_TENKAN, SIG_BELOW_KIJUN,
    SIG_FLAT_KIJUN_BEAR, SIG_CCLAMP_BEAR, SIG_E2E_BEAR,
)
ALL_SIGNALS = BULLISH_SIGNALS + BEARISH_SIGNALS

# At least one of these must fire for an entry to be recommended.
PRIMARY_BULLISH = frozenset({
    SIG_CLOUD_BREAKOUT, SIG_TK_CROSS_BULL, SIG_FLAT_KIJUN_BULL, SIG_CCLAMP_BULL,
})

REQUIRED_COLUMNS = (
    "close", "high", "low", "tenkan", "kijun",
    "cloud_top", "cloud_bottom", "senkou_a_future", "senkou_b_future",
)


@dataclass
class PatternParams:
    """Fuzzy thresholds for the 4b pattern signals. Tune against the backtest."""
    e2e_min_cloud_frac: float = 0.005     # cloud thickness >= 0.5% of price
    cclamp_depth_frac: float = 0.003      # |Tenkan - Kijun| >= 0.3% of price
    cclamp_min_bars: int = 3              # divergence must persist >= 3 bars
    flat_window: int = 5                  # Kijun "flat" measured over 5 bars
    flat_tol: float = 0.0015              # Kijun range over window <= 0.15%
    touch_tol: float = 0.0025             # within 0.25% counts as touching Kijun
    convincing_margin: float = 0.0025     # close 0.25% beyond Kijun = convincing


@dataclass
class SignalWeights:
    """Bullish confidence weights. TUNE THESE (re-run the backtest after)."""
    cloud_breakout: float = 0.6
    tk_cross: float = 0.5
    kumo_twist: float = 0.25
    flat_kijun: float = 0.55
    c_clamp: float = 0.5
    e2e: float = 0.4

    def weight_for(self, name: str) -> float:
        return {
            SIG_CLOUD_BREAKOUT: self.cloud_breakout,
            SIG_TK_CROSS_BULL: self.tk_cross,
            SIG_KUMO_TWIST_BULL: self.kumo_twist,
            SIG_FLAT_KIJUN_BULL: self.flat_kijun,
            SIG_CCLAMP_BULL: self.c_clamp,
            SIG_E2E_BULL: self.e2e,
        }.get(name, 0.0)


@dataclass
class SignalResult:
    timestamp: object
    bullish_signals: list[str]
    bearish_signals: list[str]
    confidence: float
    entry_recommended: bool
    exit_recommended: bool
    details: dict

    def summary(self) -> str:
        bull = ",".join(self.bullish_signals) or "-"
        bear = ",".join(self.bearish_signals) or "-"
        return (f"conf={self.confidence:.2f} entry={self.entry_recommended} "
                f"exit={self.exit_recommended} bull=[{bull}] bear=[{bear}]")


def signals_per_row(ich: pd.DataFrame, params: PatternParams | None = None) -> pd.DataFrame:
    """Boolean DataFrame: for every row, which signals fired. Vectorized.

    Comparisons against NaN evaluate to False, so warmup rows never fire.
    """
    if params is None:
        params = PatternParams()
    missing = [c for c in REQUIRED_COLUMNS if c not in ich.columns]
    if missing:
        raise ValueError(f"Ichimoku DataFrame missing columns: {missing}")

    close, high, low = ich["close"], ich["high"], ich["low"]
    tenkan, kijun = ich["tenkan"], ich["kijun"]
    ct, cb = ich["cloud_top"], ich["cloud_bottom"]

    def cross_above(a, b):
        return (a.shift(1) <= b.shift(1)) & (a > b)

    def cross_below(a, b):
        return (a.shift(1) >= b.shift(1)) & (a < b)

    out = pd.DataFrame(index=ich.index)

    # --- core (4a) ---------------------------------------------------------
    out[SIG_TK_CROSS_BULL] = cross_above(tenkan, kijun)
    out[SIG_CLOUD_BREAKOUT] = cross_above(close, ct)
    out[SIG_KUMO_TWIST_BULL] = cross_above(ich["senkou_a_future"], ich["senkou_b_future"])
    out[SIG_BELOW_TENKAN] = cross_below(close, tenkan)
    out[SIG_BELOW_KIJUN] = cross_below(close, kijun)

    # --- edge-to-edge (4b) -------------------------------------------------
    thick_ok = ((ct - cb) / close) >= params.e2e_min_cloud_frac
    inside = (close > cb) & (close < ct)
    out[SIG_E2E_BULL] = (close.shift(1) <= cb.shift(1)) & inside & thick_ok   # entered from below
    out[SIG_E2E_BEAR] = (close.shift(1) >= ct.shift(1)) & inside & thick_ok   # entered from above

    # --- C-clamp (4b): Tenkan/Kijun divergence, then price reclaims Tenkan --
    m = params.cclamp_min_bars
    tk_below = tenkan < kijun
    tk_above = tenkan > kijun
    gap_below = (kijun - tenkan) / close
    gap_above = (tenkan - kijun) / close
    dur_below = tk_below.astype(int).rolling(m).sum() == m
    dur_above = tk_above.astype(int).rolling(m).sum() == m
    mag_below = gap_below.rolling(m).max() >= params.cclamp_depth_frac
    mag_above = gap_above.rolling(m).max() >= params.cclamp_depth_frac
    out[SIG_CCLAMP_BULL] = (
        cross_above(close, tenkan) & tk_below & (close < kijun) & dur_below & mag_below
    )
    out[SIG_CCLAMP_BEAR] = (
        cross_below(close, tenkan) & tk_above & (close > kijun) & dur_above & mag_above
    )

    # --- flat-Kijun (4b): bounce-from-above OR break-from-below -------------
    kij_range = kijun.rolling(params.flat_window).max() - kijun.rolling(params.flat_window).min()
    flat = (kij_range / close) <= params.flat_tol
    conv_above = close >= kijun * (1 + params.convincing_margin)
    conv_below = close <= kijun * (1 - params.convincing_margin)
    touch_from_above = low <= kijun * (1 + params.touch_tol)     # wick down to Kijun
    touch_from_below = high >= kijun * (1 - params.touch_tol)    # wick up to Kijun
    break_from_below = close.shift(1) <= kijun.shift(1)
    break_from_above = close.shift(1) >= kijun.shift(1)
    out[SIG_FLAT_KIJUN_BULL] = flat & conv_above & (touch_from_above | break_from_below)
    out[SIG_FLAT_KIJUN_BEAR] = flat & conv_below & (touch_from_below | break_from_above)

    return out.fillna(False).astype(bool)


def _neutral_result(timestamp) -> SignalResult:
    return SignalResult(timestamp, [], [], 0.0, False, False, {})


def evaluate_signals(
    ich: pd.DataFrame,
    min_confidence: float,
    weights: SignalWeights | None = None,
    params: PatternParams | None = None,
) -> SignalResult:
    """Evaluate all signals on the latest completed candle."""
    if weights is None:
        weights = SignalWeights()

    has_time = "time" in ich.columns
    if len(ich) < 2:
        ts = ich.iloc[-1]["time"] if (len(ich) == 1 and has_time) else None
        return _neutral_result(ts)

    flags = signals_per_row(ich, params)
    last = flags.iloc[-1]
    fired = {name: bool(last[name]) for name in ALL_SIGNALS}

    bullish = [name for name in BULLISH_SIGNALS if fired[name]]
    bearish = [name for name in BEARISH_SIGNALS if fired[name]]

    confidence = min(1.0, sum(weights.weight_for(name) for name in bullish))
    has_primary = any(name in PRIMARY_BULLISH for name in bullish)
    entry_recommended = confidence >= min_confidence and has_primary
    exit_recommended = len(bearish) > 0

    details = {
        name: {"fired": fired[name],
               "weight": weights.weight_for(name) if name in BULLISH_SIGNALS else 0.0}
        for name in ALL_SIGNALS
    }
    ts = ich.iloc[-1]["time"] if has_time else None
    return SignalResult(ts, bullish, bearish, confidence, entry_recommended, exit_recommended, details)
