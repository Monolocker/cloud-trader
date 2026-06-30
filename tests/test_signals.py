from __future__ import annotations
import pandas as pd
import pytest
from ichibot.signals import (
    SIG_BELOW_KIJUN, SIG_BELOW_TENKAN, SIG_CCLAMP_BEAR, SIG_CCLAMP_BULL,
    SIG_CLOUD_BREAKOUT, SIG_E2E_BEAR, SIG_E2E_BULL, SIG_FLAT_KIJUN_BEAR,
    SIG_FLAT_KIJUN_BULL, SIG_KUMO_TWIST_BULL, SIG_OVEREXTENDED, SIG_TK_CROSS_BULL,
    PatternParams, SignalWeights, evaluate_signals, signals_per_row,
)

_DEF = {"close": 100.0, "high": 100.0, "low": 100.0, "tenkan": 100.0, "kijun": 100.0,
        "cloud_top": 200.0, "cloud_bottom": 190.0, "senkou_a_future": 150.0, "senkou_b_future": 150.0}


def _frame(rows):
    df = pd.DataFrame([{**_DEF, **r} for r in rows])
    df["time"] = pd.date_range("2025-01-01", periods=len(df), freq="D", tz="UTC")
    return df


def _eval(rows, min_conf=0.6, params=None):
    return evaluate_signals(_frame(rows), min_conf, params=params)


def _two(o0, o1):
    return _frame([{**_DEF, **o0}, {**_DEF, **o1}])


# --- 4a core --------------------------------------------------------------
def test_bullish_tk_cross():
    assert bool(signals_per_row(_two({"tenkan": 90, "kijun": 95}, {"tenkan": 96, "kijun": 95})).iloc[-1][SIG_TK_CROSS_BULL])


def test_cloud_breakout():
    assert bool(signals_per_row(_two({"close": 69, "cloud_top": 70}, {"close": 75, "cloud_top": 70})).iloc[-1][SIG_CLOUD_BREAKOUT])


def test_kumo_twist():
    assert bool(signals_per_row(_two({"senkou_a_future": 90, "senkou_b_future": 95}, {"senkou_a_future": 96, "senkou_b_future": 95})).iloc[-1][SIG_KUMO_TWIST_BULL])


def test_below_tenkan():
    assert bool(signals_per_row(_two({"close": 95, "tenkan": 90}, {"close": 85, "tenkan": 90})).iloc[-1][SIG_BELOW_TENKAN])


def test_below_kijun():
    assert bool(signals_per_row(_two({"close": 95, "kijun": 90}, {"close": 85, "kijun": 90})).iloc[-1][SIG_BELOW_KIJUN])


def test_no_signal_quiet_frame():
    r = _eval([_DEF, _DEF])
    assert r.bullish_signals == [] and r.bearish_signals == []


def test_breakout_alone_meets_threshold():
    r = _eval([{"close": 69, "cloud_top": 70}, {"close": 75, "cloud_top": 70}])
    assert r.confidence == pytest.approx(0.6) and r.entry_recommended


def test_tk_cross_alone_below_threshold():
    r = _eval([{"tenkan": 90, "kijun": 95}, {"tenkan": 96, "kijun": 95}])
    assert r.confidence == pytest.approx(0.5) and not r.entry_recommended


def test_confidence_caps_at_one():
    r = _eval([{"close": 69, "cloud_top": 70, "tenkan": 90, "kijun": 95, "senkou_a_future": 90, "senkou_b_future": 95},
               {"close": 75, "cloud_top": 70, "tenkan": 96, "kijun": 95, "senkou_a_future": 96, "senkou_b_future": 95}])
    assert r.confidence == pytest.approx(1.0) and r.entry_recommended


def test_exit_on_any_bearish():
    assert _eval([{"close": 95, "tenkan": 90}, {"close": 85, "tenkan": 90}]).exit_recommended


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        signals_per_row(pd.DataFrame([{"close": 1.0}]))


def test_too_short_is_neutral():
    r = evaluate_signals(_frame([_DEF]), 0.6)
    assert not r.entry_recommended and not r.exit_recommended


def test_weights_are_tunable():
    r = evaluate_signals(_frame([{"close": 69, "cloud_top": 70}, {"close": 75, "cloud_top": 70}]),
                         0.6, weights=SignalWeights(cloud_breakout=0.9))
    assert r.confidence == pytest.approx(0.9)


def test_entry_needs_primary_not_just_confidence():
    assert not _eval([{"senkou_a_future": 90, "senkou_b_future": 95}, {"senkou_a_future": 96, "senkou_b_future": 95}], min_conf=0.2).entry_recommended


def test_two_candle_frame_fires_no_patterns():
    flags = signals_per_row(_two({"close": 69, "cloud_top": 70}, {"close": 75, "cloud_top": 70})).iloc[-1]
    for n in (SIG_E2E_BULL, SIG_CCLAMP_BULL, SIG_FLAT_KIJUN_BULL, SIG_E2E_BEAR,
              SIG_CCLAMP_BEAR, SIG_FLAT_KIJUN_BEAR, SIG_OVEREXTENDED):
        assert not bool(flags[n])


# --- 4b patterns ----------------------------------------------------------
def test_e2e_bull_enters_cloud_from_below():
    assert bool(signals_per_row(_frame([{"close": 40, "cloud_bottom": 50, "cloud_top": 70},
                                        {"close": 60, "cloud_bottom": 50, "cloud_top": 70}])).iloc[-1][SIG_E2E_BULL])


def test_e2e_bear_enters_cloud_from_above():
    assert bool(signals_per_row(_frame([{"close": 80, "cloud_bottom": 50, "cloud_top": 70},
                                        {"close": 60, "cloud_bottom": 50, "cloud_top": 70}])).iloc[-1][SIG_E2E_BEAR])


def test_e2e_rejects_thin_cloud():
    assert not bool(signals_per_row(_frame([{"close": 40, "cloud_bottom": 70.0, "cloud_top": 70.1},
                                            {"close": 70.05, "cloud_bottom": 70.0, "cloud_top": 70.1}])).iloc[-1][SIG_E2E_BULL])


def test_cclamp_bull():
    rows = [{"close": 97.0, "tenkan": 99, "kijun": 100}, {"close": 97.0, "tenkan": 99, "kijun": 100},
            {"close": 98.0, "tenkan": 99, "kijun": 100}, {"close": 99.5, "tenkan": 99, "kijun": 100}]
    assert bool(signals_per_row(_frame(rows)).iloc[-1][SIG_CCLAMP_BULL])


def test_cclamp_bull_rejected_too_brief():
    rows = [{"close": 101.0, "tenkan": 101, "kijun": 100}, {"close": 101.0, "tenkan": 101, "kijun": 100},
            {"close": 98.0, "tenkan": 99, "kijun": 100}, {"close": 99.5, "tenkan": 99, "kijun": 100}]
    assert not bool(signals_per_row(_frame(rows)).iloc[-1][SIG_CCLAMP_BULL])


def test_cclamp_bear():
    rows = [{"close": 103.0, "tenkan": 101, "kijun": 100}, {"close": 103.0, "tenkan": 101, "kijun": 100},
            {"close": 102.0, "tenkan": 101, "kijun": 100}, {"close": 100.5, "tenkan": 101, "kijun": 100}]
    assert bool(signals_per_row(_frame(rows)).iloc[-1][SIG_CCLAMP_BEAR])


def test_flat_kijun_bull_break_from_below():
    rows = [dict({"close": 99.0, "kijun": 100, "tenkan": 100, "low": 98.5}) for _ in range(5)] + \
           [{"close": 100.5, "kijun": 100, "tenkan": 100, "low": 99.0}]
    assert bool(signals_per_row(_frame(rows)).iloc[-1][SIG_FLAT_KIJUN_BULL])


def test_flat_kijun_bull_bounce_from_above():
    rows = [dict({"close": 101.0, "kijun": 100, "tenkan": 100, "low": 100.6}) for _ in range(5)] + \
           [{"close": 101.0, "kijun": 100, "tenkan": 100, "low": 100.0}]
    assert bool(signals_per_row(_frame(rows)).iloc[-1][SIG_FLAT_KIJUN_BULL])


def test_flat_kijun_bear_break_from_above():
    rows = [dict({"close": 101.0, "kijun": 100, "tenkan": 100, "high": 101.5}) for _ in range(5)] + \
           [{"close": 99.5, "kijun": 100, "tenkan": 100, "high": 100.5}]
    assert bool(signals_per_row(_frame(rows)).iloc[-1][SIG_FLAT_KIJUN_BEAR])


def test_flat_kijun_requires_flatness():
    rows = [{"close": 99.0, "kijun": 90, "tenkan": 90, "low": 98}, {"close": 99.0, "kijun": 93, "tenkan": 93, "low": 98},
            {"close": 99.0, "kijun": 96, "tenkan": 96, "low": 98}, {"close": 99.0, "kijun": 99, "tenkan": 99, "low": 98},
            {"close": 101.0, "kijun": 100, "tenkan": 100, "low": 99}, {"close": 102.0, "kijun": 101, "tenkan": 101, "low": 100}]
    assert not bool(signals_per_row(_frame(rows)).iloc[-1][SIG_FLAT_KIJUN_BULL])


def test_pattern_entry_flows_through_evaluate():
    rows = [dict({"close": 99.0, "kijun": 100, "tenkan": 100, "low": 98.5}) for _ in range(5)] + \
           [{"close": 100.5, "kijun": 100, "tenkan": 100, "low": 99.0}]
    r = evaluate_signals(_frame(rows), 0.6)
    assert SIG_FLAT_KIJUN_BULL in r.bullish_signals
    assert r.confidence == pytest.approx(0.55) and not r.entry_recommended


def test_params_are_tunable():
    rows = [dict({"close": 99.0, "kijun": 100, "tenkan": 100, "low": 98.5}) for _ in range(5)] + \
           [{"close": 100.5, "kijun": 100, "tenkan": 100, "low": 99.0}]
    assert bool(signals_per_row(_frame(rows), PatternParams(flat_tol=0.0)).iloc[-1][SIG_FLAT_KIJUN_BULL])
    assert not bool(signals_per_row(_frame(rows), PatternParams(convincing_margin=0.02)).iloc[-1][SIG_FLAT_KIJUN_BULL])


# --- overextension exit ---------------------------------------------------
_OE = PatternParams(overext_lookback=5, overext_percentile=0.90)


def test_overextension_fires_on_extreme_gap():
    # gap ~1% for 5 bars, then spikes to 20% -> exceeds the 90th pct of its history
    rows = [{"tenkan": 101, "kijun": 100} for _ in range(5)] + [{"tenkan": 120, "kijun": 100}]
    assert bool(signals_per_row(_frame(rows), _OE).iloc[-1][SIG_OVEREXTENDED])


def test_overextension_quiet_when_gap_steady():
    # constant gap -> current never strictly exceeds its own percentile -> no fire
    rows = [{"tenkan": 101, "kijun": 100} for _ in range(8)]
    assert not bool(signals_per_row(_frame(rows), _OE).iloc[-1][SIG_OVEREXTENDED])


def test_overextension_ignores_downside_gap():
    # tenkan below kijun (negative gap) -> never an upside overextension
    rows = [{"tenkan": 99, "kijun": 100} for _ in range(5)] + [{"tenkan": 80, "kijun": 100}]
    assert not bool(signals_per_row(_frame(rows), _OE).iloc[-1][SIG_OVEREXTENDED])


def test_overextension_toggle_off():
    rows = [{"tenkan": 101, "kijun": 100} for _ in range(5)] + [{"tenkan": 120, "kijun": 100}]
    off = PatternParams(overext_lookback=5, overext_percentile=0.90, use_overextension_exit=False)
    assert not bool(signals_per_row(_frame(rows), off).iloc[-1][SIG_OVEREXTENDED])


def test_overextension_triggers_exit_recommended():
    rows = [{"tenkan": 101, "kijun": 100} for _ in range(5)] + [{"tenkan": 120, "kijun": 100}]
    r = evaluate_signals(_frame(rows), 0.6, params=_OE)
    assert r.exit_recommended and SIG_OVEREXTENDED in r.bearish_signals