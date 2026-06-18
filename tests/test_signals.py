"""Tests for the five core signals (Milestone 4a). Network-free.

Signal logic operates on Ichimoku columns, small DataFrames are built with the
relevant columns set directly to force each crossing -- no need to reverse-engineer
OHLC that produces a given cross.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ichibot.signals import (
    SIG_BELOW_KIJUN,
    SIG_BELOW_TENKAN,
    SIG_CLOUD_BREAKOUT,
    SIG_KUMO_TWIST_BULL,
    SIG_TK_CROSS_BULL,
    SignalWeights,
    evaluate_signals,
    signals_per_row,
)

# Baseline column values for one "boring" candle (no crossings). Each test
# overrides prev/curr values for the columns relevant to its signal.
_BASE = {
    "close": 100.0,
    "tenkan": 90.0,
    "kijun": 80.0,
    "cloud_top": 70.0,
    "senkou_a_future": 60.0,
    "senkou_b_future": 50.0,
}


def _two_rows(prev_over: dict, curr_over: dict) -> pd.DataFrame:
    prev = {**_BASE, **prev_over}
    curr = {**_BASE, **curr_over}
    df = pd.DataFrame([prev, curr])
    df["time"] = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    return df


def test_bullish_tk_cross_fires():
    df = _two_rows({"tenkan": 79.0, "kijun": 80.0}, {"tenkan": 85.0, "kijun": 80.0})
    res = evaluate_signals(df, min_confidence=0.6)
    assert SIG_TK_CROSS_BULL in res.bullish_signals


def test_no_tk_cross_when_already_above():
    df = _two_rows({"tenkan": 85.0, "kijun": 80.0}, {"tenkan": 88.0, "kijun": 80.0})
    res = evaluate_signals(df, min_confidence=0.6)
    assert SIG_TK_CROSS_BULL not in res.bullish_signals


def test_cloud_breakout_fires():
    df = _two_rows({"close": 69.0, "cloud_top": 70.0}, {"close": 75.0, "cloud_top": 70.0})
    res = evaluate_signals(df, min_confidence=0.6)
    assert SIG_CLOUD_BREAKOUT in res.bullish_signals


def test_kumo_twist_fires():
    df = _two_rows(
        {"senkou_a_future": 49.0, "senkou_b_future": 50.0},
        {"senkou_a_future": 55.0, "senkou_b_future": 50.0},
    )
    res = evaluate_signals(df, min_confidence=0.6)
    assert SIG_KUMO_TWIST_BULL in res.bullish_signals


def test_below_tenkan_fires_and_recommends_exit():
    df = _two_rows({"close": 95.0, "tenkan": 90.0}, {"close": 85.0, "tenkan": 90.0})
    res = evaluate_signals(df, min_confidence=0.6)
    assert SIG_BELOW_TENKAN in res.bearish_signals
    assert res.exit_recommended is True


def test_below_kijun_fires():
    df = _two_rows({"close": 95.0, "kijun": 80.0}, {"close": 75.0, "kijun": 80.0})
    res = evaluate_signals(df, min_confidence=0.6)
    assert SIG_BELOW_KIJUN in res.bearish_signals


def test_confidence_sums_and_caps_at_one():
    df = _two_rows(
        {"close": 69.0, "cloud_top": 70.0, "tenkan": 79.0, "kijun": 80.0,
         "senkou_a_future": 49.0, "senkou_b_future": 50.0},
        {"close": 75.0, "cloud_top": 70.0, "tenkan": 85.0, "kijun": 80.0,
         "senkou_a_future": 55.0, "senkou_b_future": 50.0},
    )
    res = evaluate_signals(df, min_confidence=0.6)
    assert res.confidence == 1.0
    assert res.entry_recommended is True


def test_breakout_alone_meets_threshold():
    df = _two_rows({"close": 69.0, "cloud_top": 70.0}, {"close": 75.0, "cloud_top": 70.0})
    res = evaluate_signals(df, min_confidence=0.6)
    assert res.confidence == pytest.approx(0.6)
    assert res.entry_recommended is True


def test_tk_cross_alone_below_threshold():
    df = _two_rows({"tenkan": 79.0, "kijun": 80.0}, {"tenkan": 85.0, "kijun": 80.0})
    res = evaluate_signals(df, min_confidence=0.6)
    assert res.confidence == pytest.approx(0.5)
    assert res.entry_recommended is False


def test_entry_requires_a_primary_even_if_threshold_met():
    weights = SignalWeights(kumo_twist=0.7)
    df = _two_rows(
        {"senkou_a_future": 49.0, "senkou_b_future": 50.0},
        {"senkou_a_future": 55.0, "senkou_b_future": 50.0},
    )
    res = evaluate_signals(df, min_confidence=0.6, weights=weights)
    assert res.confidence == pytest.approx(0.7)
    assert res.entry_recommended is False


def test_exit_not_gated_by_confidence():
    df = _two_rows({"close": 95.0, "kijun": 80.0}, {"close": 75.0, "kijun": 80.0})
    res = evaluate_signals(df, min_confidence=0.6)
    assert res.confidence == 0.0
    assert res.exit_recommended is True


def test_too_few_rows_is_neutral():
    one = pd.DataFrame([_BASE])
    one["time"] = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC")
    res = evaluate_signals(one, min_confidence=0.6)
    assert res.bullish_signals == [] and res.bearish_signals == []
    assert res.entry_recommended is False and res.exit_recommended is False


def test_missing_columns_raises():
    df = _two_rows({}, {}).drop(columns=["cloud_top"])
    with pytest.raises(ValueError):
        evaluate_signals(df, min_confidence=0.6)


def test_signals_per_row_finds_cross_at_right_index():
    rows = [
        {**_BASE, "tenkan": 70.0, "kijun": 80.0},
        {**_BASE, "tenkan": 75.0, "kijun": 80.0},
        {**_BASE, "tenkan": 85.0, "kijun": 80.0},
        {**_BASE, "tenkan": 88.0, "kijun": 80.0},
    ]
    df = pd.DataFrame(rows)
    df["time"] = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    flags = signals_per_row(df)
    assert flags[SIG_TK_CROSS_BULL].tolist() == [False, False, True, False]


def test_evaluate_matches_signals_per_row_last():
    df = _two_rows({"tenkan": 79.0, "kijun": 80.0}, {"tenkan": 85.0, "kijun": 80.0})
    res = evaluate_signals(df, min_confidence=0.6)
    last = signals_per_row(df).iloc[-1]
    assert bool(last[SIG_TK_CROSS_BULL]) is (SIG_TK_CROSS_BULL in res.bullish_signals)