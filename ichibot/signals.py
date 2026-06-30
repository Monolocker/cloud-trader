"""Signal detection for the Ichimoku strategy.

4a: five core crossing signals.
4b: three pattern signals (edge-to-edge, C-clamp, flat-Kijun), bullish entry + bearish exit.
Exit milestone: a TK overextension exit strategy. Fires when the gap between Tenkan
and Kijun is wider than its own recent history, i.e. price has stretched
too far above equilibrium. Exit-only; sells into strength before the structure breaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SIG_TK_CROSS_BULL = "bullish_tk_cross"
SIG_CLOUD_BREAKOUT = "price_breakout_above_cloud"
SIG_KUMO_TWIST_BULL = "bullish_kumo_twist"
SIG_BELOW_TENKAN = "price_below_tenkan"
SIG_BELOW_KIJUN = "price_below_kijun"

SIG_E2E_BULL = "edge_to_edge_bull"
SIG_CCLAMP_BULL = "c_clamp_bull"
SIG_FLAT_KIJUN_BULL = "flat_kijun_bull"
SIG_E2E_BEAR = "edge_to_edge_bear"
SIG_CCLAMP_BEAR = "c_clamp_bear"
SIG_FLAT_KIJUN_BEAR = "flat_kijun_bear"

SIG_OVEREXTENDED = "tenkan_kijun_overextended"   # exit-only overextension signal

BULLISH_SIGNALS = (
    SIG_CLOUD_BREAKOUT, SIG_TK_CROSS_BULL, SIG_KUMO_TWIST_BULL,
    SIG_FLAT_KIJUN_BULL, SIG_CCLAMP_BULL, SIG_E2E_BULL,
)
BEARISH_SIGNALS = (
    SIG_BELOW_TENKAN, SIG_BELOW_KIJUN,
    SIG_FLAT_KIJUN_BEAR, SIG_CCLAMP_BEAR, SIG_E2E_BEAR,
    SIG_OVEREXTENDED,
)
ALL_SIGNALS = BULLISH_SIGNALS + BEARISH_SIGNALS

PRIMARY_BULLISH = frozenset({
    SIG_CLOUD_BREAKOUT, SIG_TK_CROSS_BULL, SIG_FLAT_KIJUN_BULL, SIG_CCLAMP_BULL,
})

REQUIRED_COLUMNS = (
    "close", "high", "low", "tenkan", "kijun",
    "cloud_top", "cloud_bottom", "senkou_a_future", "senkou_b_future",
)


@dataclass
class PatternParams:
    """Fuzzy thresholds. Tune against the backtest."""
    e2e_min_cloud_frac: float = 0.005
    cclamp_depth_frac: float = 0.003
    cclamp_min_bars: int = 3
    flat_window: int = 5
    flat_tol: float = 0.0015
    touch_tol: float = 0.0025
    convincing_margin: float = 0.0025
    # --- overextension exit (Tenkan-Kijun gap percentile) ---
    use_overextension_exit: bool = False   # toggle for A/B testing
    overext_lookback: int = 60            # window for "recent history"
    overext_percentile: float = 0.90      # exit when gap exceeds this percentile of its history


@dataclass
class SignalWeights:
    cloud_breakout: float = 0.6
    tk_cross: float = 0.5
    kumo_twist: float = 0.25
    flat_kijun: float = 0.55
    c_clamp: float = 0.5
    e2e: float = 0.4

    def weight_for(self, name: str) -> float:
        return {
            SIG_CLOUD_BREAKOUT: self.cloud_breakout, SIG_TK_CROSS_BULL: self.tk_cross,
            SIG_KUMO_TWIST_BULL: self.kumo_twist, SIG_FLAT_KIJUN_BULL: self.flat_kijun,
            SIG_CCLAMP_BULL: self.c_clamp, SIG_E2E_BULL: self.e2e,
        }.get(name, 0.0)


@dataclass
class SignalResult:
    timestamp: object
    bullish_signals: list
    bearish_signals: list
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
    out[SIG_TK_CROSS_BULL] = cross_above(tenkan, kijun)
    out[SIG_CLOUD_BREAKOUT] = cross_above(close, ct)
    out[SIG_KUMO_TWIST_BULL] = cross_above(ich["senkou_a_future"], ich["senkou_b_future"])
    out[SIG_BELOW_TENKAN] = cross_below(close, tenkan)
    out[SIG_BELOW_KIJUN] = cross_below(close, kijun)

    thick_ok = ((ct - cb) / close) >= params.e2e_min_cloud_frac
    inside = (close > cb) & (close < ct)
    out[SIG_E2E_BULL] = (close.shift(1) <= cb.shift(1)) & inside & thick_ok
    out[SIG_E2E_BEAR] = (close.shift(1) >= ct.shift(1)) & inside & thick_ok

    m = params.cclamp_min_bars
    tk_below = tenkan < kijun
    tk_above = tenkan > kijun
    gap_below = (kijun - tenkan) / close
    gap_above = (tenkan - kijun) / close
    out[SIG_CCLAMP_BULL] = (cross_above(close, tenkan) & tk_below & (close < kijun)
                            & (tk_below.astype(int).rolling(m).sum() == m)
                            & (gap_below.rolling(m).max() >= params.cclamp_depth_frac))
    out[SIG_CCLAMP_BEAR] = (cross_below(close, tenkan) & tk_above & (close > kijun)
                            & (tk_above.astype(int).rolling(m).sum() == m)
                            & (gap_above.rolling(m).max() >= params.cclamp_depth_frac))

    kij_range = kijun.rolling(params.flat_window).max() - kijun.rolling(params.flat_window).min()
    flat = (kij_range / close) <= params.flat_tol
    out[SIG_FLAT_KIJUN_BULL] = (flat & (close >= kijun * (1 + params.convincing_margin))
                                & ((low <= kijun * (1 + params.touch_tol)) | (close.shift(1) <= kijun.shift(1))))
    out[SIG_FLAT_KIJUN_BEAR] = (flat & (close <= kijun * (1 - params.convincing_margin))
                                & ((high >= kijun * (1 - params.touch_tol)) | (close.shift(1) >= kijun.shift(1))))

    # --- Tenkan-Kijun overextension exit -----------------------------------
    # gap as a fraction of Kijun, so "Tenkan 15% above Kijun" == 0.15. Fire when the
    # current gap is strictly above the Nth percentile of its own recent history AND
    # positive (an UPSIDE stretch). Strict '>' so a steady gap doesn't fire every bar.
    gap = (tenkan - kijun) / kijun
    roll_q = gap.rolling(params.overext_lookback).quantile(params.overext_percentile)
    overext = (gap > 0) & (gap > roll_q)
    if not params.use_overextension_exit:
        overext = pd.Series(False, index=ich.index)
    out[SIG_OVEREXTENDED] = overext

    return out.fillna(False).astype(bool)


def evaluate_signals(ich, min_confidence, weights=None, params=None):
    if weights is None:
        weights = SignalWeights()
    has_time = "time" in ich.columns
    if len(ich) < 2:
        ts = ich.iloc[-1]["time"] if (len(ich) == 1 and has_time) else None
        return SignalResult(ts, [], [], 0.0, False, False, {})
    flags = signals_per_row(ich, params)
    last = flags.iloc[-1]
    fired = {name: bool(last[name]) for name in ALL_SIGNALS}
    bullish = [n for n in BULLISH_SIGNALS if fired[n]]
    bearish = [n for n in BEARISH_SIGNALS if fired[n]]
    confidence = min(1.0, sum(weights.weight_for(n) for n in bullish))
    entry = confidence >= min_confidence and any(n in PRIMARY_BULLISH for n in bullish)
    ts = ich.iloc[-1]["time"] if has_time else None
    return SignalResult(ts, bullish, bearish, confidence, entry, len(bearish) > 0, {})