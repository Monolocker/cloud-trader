"""Temporary validation utility (not part of the ichibot package).

Cross-checks ichibot.ichimoku.compute_ichimoku (pandas: rolling + shift) against
an naive from-scratch implementation (plain Python loops) on the exact
Hyperliquid candles the bot fetches. the indicator math is correct upon agreement 
on the relevant trade data. No chart required. File not imported anywhere
"""

from __future__ import annotations

import math

import pandas as pd

from ichibot.config import load_config
from ichibot.ichimoku import compute_ichimoku
from ichibot.market_data import HyperliquidData


def naive_ichimoku(df: pd.DataFrame, conv: int, base: int, spanb: int, disp: int) -> dict:
    """Independent reference implementation using only plain Python loops."""
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    n = len(df)

    def midpoint(i: int, period: int):
        if i < period - 1:
            return math.nan
        window_hi = max(highs[i - period + 1 : i + 1])
        window_lo = min(lows[i - period + 1 : i + 1])
        return (window_hi + window_lo) / 2.0

    tenkan = [midpoint(i, conv) for i in range(n)]
    kijun = [midpoint(i, base) for i in range(n)]
    span_a_raw = [
        (tenkan[i] + kijun[i]) / 2.0 if not (math.isnan(tenkan[i]) or math.isnan(kijun[i])) else math.nan
        for i in range(n)
    ]
    span_b_raw = [midpoint(i, spanb) for i in range(n)]

    senkou_a = [span_a_raw[i - disp] if i - disp >= 0 else math.nan for i in range(n)]
    senkou_b = [span_b_raw[i - disp] if i - disp >= 0 else math.nan for i in range(n)]
    chikou = [closes[i + disp] if i + disp < n else math.nan for i in range(n)]

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }


def _series_match(a: pd.Series, b_list: list, tol: float = 1e-9) -> bool:
    """True if two series agree everywhere (NaN matches NaN)."""
    if len(a) != len(b_list):
        return False
    for x, y in zip(a.tolist(), b_list):
        x_nan, y_nan = (x != x), (y != y)  # NaN check without imports
        if x_nan and y_nan:
            continue
        if x_nan != y_nan:
            return False
        if abs(x - y) > tol:
            return False
    return True


def check(df: pd.DataFrame, conv: int, base: int, spanb: int, disp: int) -> dict:
    bot = compute_ichimoku(df, conv, base, spanb, disp)
    ref = naive_ichimoku(df, conv, base, spanb, disp)
    return {line: _series_match(bot[line], ref[line]) for line in ref}


if __name__ == "__main__":
    cfg = load_config("Config.yaml", ".env")
    ic = cfg.ichimoku
    data = HyperliquidData()
    print(f"Settings: conv={ic.conversion_periods} base={ic.base_periods} "
          f"spanB={ic.span_b_periods} disp={ic.displacement}\n")
    all_ok = True
    for coin in cfg.trading.markets:
        df = data.fetch_daily(coin, lookback_days=200,
                              drop_incomplete=cfg.trading.only_completed_candles)
        results = check(df, ic.conversion_periods, ic.base_periods,
                        ic.span_b_periods, ic.displacement)
        ok = all(results.values())
        all_ok = all_ok and ok
        flags = " ".join(f"{k}={'OK' if v else 'MISMATCH'}" for k, v in results.items())
        print(f"{coin:6s} ({len(df)} candles)  ->  {'PASS' if ok else 'FAIL'}   [{flags}]")
    print("\n" + ("ALL PASS: pandas implementation matches the independent recompute."
                  if all_ok else "FAIL: see mismatches above."))
