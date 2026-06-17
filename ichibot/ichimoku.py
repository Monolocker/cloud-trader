"""Ichimoku Cloud indicator computation.

Pure functions over candle DataFrames (the output of market_data). No network
calls and no config import: the periods are passed in as plain
integers so this module is trivial to unit-test.

Note on displacement:
  - The two Senkou spans (the "cloud") are plotted FORWARD in time. The cloud
    sitting under today's price was computed `displacement` bars ago. So to get
    a column whose value at today's row is the cloud beneath today's price, we
    shift the computed span forward: span.shift(displacement).
  - The Chikou ("lagging") span is the opposite: today's close plotted
    `displacement` bars BACK, i.e. close.shift(-displacement).
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["high", "low", "close"]


class IchimokuError(Exception):
    """Raised when Ichimoku cannot be computed (bad input or parameters)."""


def min_required_candles(span_b_periods: int = 120, displacement: int = 30) -> int:
    """Smallest number of candles needed for at least one fully-formed cloud row."""
    return span_b_periods + displacement


def _midpoint(high: pd.Series, low: pd.Series, periods: int) -> pd.Series:
    """(highest high + lowest low) / 2 over a rolling window of `periods` bars."""
    highest = high.rolling(window=periods).max()
    lowest = low.rolling(window=periods).min()
    return (highest + lowest) / 2.0


def compute_ichimoku(
    df: pd.DataFrame,
    conversion_periods: int = 20,
    base_periods: int = 60,
    span_b_periods: int = 120,
    displacement: int = 30,
) -> pd.DataFrame:
    """Return a COPY of `df` with Ichimoku columns added.

    Added columns:
      tenkan           Conversion line: midpoint over conversion_periods.
      kijun            Base line: midpoint over base_periods.
      senkou_a         Leading Span A, shifted FORWARD by displacement so it
                       aligns with the current candle (cloud beneath today).
      senkou_b         Leading Span B, shifted forward by displacement.
      senkou_a_future  Leading Span A as computed at this candle (the cloud that
                       will be plotted `displacement` bars ahead) -> twist checks.
      senkou_b_future  Leading Span B as computed at this candle.
      chikou           Lagging span: close shifted BACK by displacement.
      cloud_top        max(senkou_a, senkou_b) at the current candle.
      cloud_bottom     min(senkou_a, senkou_b) at the current candle.

    Early rows are NaN until enough history exists (expected). Rows are never
    dropped, so the output keeps the same length and order as the input.
    """
    for periods, name in (
        (conversion_periods, "conversion_periods"),
        (base_periods, "base_periods"),
        (span_b_periods, "span_b_periods"),
        (displacement, "displacement"),
    ):
        if not isinstance(periods, int) or periods <= 0:
            raise IchimokuError(f"{name} must be a positive integer (got {periods!r})")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IchimokuError(f"DataFrame is missing required columns: {missing}")

    out = df.copy()
    high, low, close = out["high"], out["low"], out["close"]

    out["tenkan"] = _midpoint(high, low, conversion_periods)
    out["kijun"] = _midpoint(high, low, base_periods)

    span_a_raw = (out["tenkan"] + out["kijun"]) / 2.0
    span_b_raw = _midpoint(high, low, span_b_periods)

    # Forward-shifted: aligned under the current price (the cloud you stand on).
    out["senkou_a"] = span_a_raw.shift(displacement)
    out["senkou_b"] = span_b_raw.shift(displacement)

    # Un-shifted leading values: the cloud projected ahead (for twist detection).
    out["senkou_a_future"] = span_a_raw
    out["senkou_b_future"] = span_b_raw

    # Lagging span.
    out["chikou"] = close.shift(-displacement)

    out["cloud_top"] = out[["senkou_a", "senkou_b"]].max(axis=1)
    out["cloud_bottom"] = out[["senkou_a", "senkou_b"]].min(axis=1)

    return out


def cloud_position(row: pd.Series) -> str:
    """Describe where one candle's close sits relative to its cloud.

    Returns 'ABOVE_CLOUD', 'BELOW_CLOUD', 'IN_CLOUD', or 'UNKNOWN' (not enough
    history). This is descriptive only -- it is NOT a trade signal.
    """
    close = row["close"]
    top, bottom = row["cloud_top"], row["cloud_bottom"]
    if pd.isna(top) or pd.isna(bottom):
        return "UNKNOWN"
    if close > top:
        return "ABOVE_CLOUD"
    if close < bottom:
        return "BELOW_CLOUD"
    return "IN_CLOUD"