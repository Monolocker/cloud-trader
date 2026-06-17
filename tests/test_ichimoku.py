"""Tests for Ichimoku computation. Network-free, with hand-verified numbers.

Uses tiny windows (conversion=2, base=3, span_b=4, displacement=1) so every
expected value can be computed by hand from the input series below.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ichibot.ichimoku import (
    IchimokuError,
    cloud_position,
    compute_ichimoku,
    min_required_candles,
)

# Hand-checkable series (5 candles).
HIGHS = [10, 12, 11, 14, 13]
LOWS = [5, 6, 4, 7, 8]
CLOSES = [8, 9, 7, 12, 11]


def _df(highs, lows, closes):
    n = len(highs)
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": [float(c) for c in closes],
            "high": [float(h) for h in highs],
            "low": [float(l) for l in lows],
            "close": [float(c) for c in closes],
            "volume": [1.0] * n,
        }
    )


def test_tenkan_values():
    out = compute_ichimoku(_df(HIGHS, LOWS, CLOSES), 2, 3, 4, 1)
    # tenkan (period 2) = (max two highs + min two lows) / 2
    assert math.isnan(out["tenkan"].iloc[0])
    assert out["tenkan"].iloc[1] == 8.5    # (max(10,12)+min(5,6))/2 = (12+5)/2
    assert out["tenkan"].iloc[4] == 10.5   # (max(14,13)+min(7,8))/2 = (14+7)/2


def test_kijun_values():
    out = compute_ichimoku(_df(HIGHS, LOWS, CLOSES), 2, 3, 4, 1)
    # kijun (period 3) = (max three highs + min three lows) / 2
    assert out["kijun"].iloc[2] == 8.0     # (max(10,12,11)+min(5,6,4))/2 = (12+4)/2
    assert out["kijun"].iloc[4] == 9.0     # (max(11,14,13)+min(4,7,8))/2 = (14+4)/2


def test_senkou_a_is_shifted_forward():
    out = compute_ichimoku(_df(HIGHS, LOWS, CLOSES), 2, 3, 4, 1)
    # span_a_raw = (tenkan+kijun)/2 ; at idx2 = (8.0+8.0)/2 = 8.0, idx3=(9.0+9.0)/2=9.0
    # senkou_a is that shifted forward by 1, so senkou_a[3]=8.0, senkou_a[4]=9.0
    assert out["senkou_a"].iloc[3] == 8.0
    assert out["senkou_a"].iloc[4] == 9.0


def test_chikou_is_shifted_back():
    out = compute_ichimoku(_df(HIGHS, LOWS, CLOSES), 2, 3, 4, 1)
    # chikou = close shifted back by 1: chikou[0]=close[1]=9, chikou[4]=NaN
    assert out["chikou"].iloc[0] == 9.0
    assert math.isnan(out["chikou"].iloc[4])


def test_cloud_bounds_are_min_max_of_spans():
    out = compute_ichimoku(_df(HIGHS, LOWS, CLOSES), 2, 3, 4, 1)
    row = out.iloc[4]
    assert row["cloud_top"] == max(row["senkou_a"], row["senkou_b"])
    assert row["cloud_bottom"] == min(row["senkou_a"], row["senkou_b"])


def test_length_and_columns_preserved():
    df = _df(HIGHS, LOWS, CLOSES)
    out = compute_ichimoku(df, 2, 3, 4, 1)
    assert len(out) == len(df)
    for col in ("tenkan", "kijun", "senkou_a", "senkou_b", "chikou", "cloud_top", "cloud_bottom"):
        assert col in out.columns
    # input is not mutated
    assert "tenkan" not in df.columns


def test_bad_period_raises():
    with pytest.raises(IchimokuError):
        compute_ichimoku(_df(HIGHS, LOWS, CLOSES), 0, 3, 4, 1)


def test_missing_column_raises():
    df = _df(HIGHS, LOWS, CLOSES).drop(columns=["high"])
    with pytest.raises(IchimokuError):
        compute_ichimoku(df, 2, 3, 4, 1)


def test_min_required_candles_default():
    assert min_required_candles(120, 30) == 150


def test_cloud_position_labels():
    assert cloud_position(pd.Series({"close": 10.0, "cloud_top": 8.0, "cloud_bottom": 5.0})) == "ABOVE_CLOUD"
    assert cloud_position(pd.Series({"close": 3.0, "cloud_top": 8.0, "cloud_bottom": 5.0})) == "BELOW_CLOUD"
    assert cloud_position(pd.Series({"close": 6.0, "cloud_top": 8.0, "cloud_bottom": 5.0})) == "IN_CLOUD"
    assert cloud_position(pd.Series({"close": 6.0, "cloud_top": float("nan"), "cloud_bottom": 5.0})) == "UNKNOWN"