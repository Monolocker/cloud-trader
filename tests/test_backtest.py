"""Tests for the backtest module. Network-free."""

from __future__ import annotations
import json
import logging, re, types
from datetime import datetime
import pandas as pd
import pytest
from ichibot.backtest import (Backtester, Trade, compute_metrics, max_drawdown,
                              replay_history, signal_attribution, _run_filename, save_results)
from ichibot.risk import RiskManager

log = logging.getLogger("ichibot.test")


def test_max_drawdown_basic():
    assert max_drawdown([1000, 1100, 900, 1000]) == pytest.approx(200 / 1100)


def test_max_drawdown_monotonic_is_zero():
    assert max_drawdown([1000, 1010, 1050, 1200]) == 0.0


def _trade(pnl, pnl_pct, bars=3):
    return Trade("X", "d1", 100.0, "d2", 100.0 + pnl, 1.0, pnl, pnl_pct, bars, "stop_loss")


def test_compute_metrics_basic():
    m = compute_metrics([_trade(10, 10.0), _trade(-5, -5.0), _trade(20, 20.0)],
                        [1000.0, 1010.0, 1005.0, 1025.0], 1000.0)
    assert m["trades"] == 3 and m["win_rate"] == pytest.approx(2 / 3)
    assert m["total_return_pct"] == pytest.approx(2.5) and m["profit_factor"] == pytest.approx(30 / 5)


def test_compute_metrics_empty():
    m = compute_metrics([], [1000.0], 1000.0)
    assert m["trades"] == 0 and m["profit_factor"] == 0.0


def _ich(closes, tenkan=90.0, kijun=85.0, cloud_top=100.0, cloud_bottom=80.0, a_fut=95.0, b_fut=90.0):
    rows = [{"close": float(c), "high": float(c), "low": float(c), "tenkan": tenkan, "kijun": kijun,
             "cloud_top": cloud_top, "cloud_bottom": cloud_bottom,
             "senkou_a_future": a_fut, "senkou_b_future": b_fut} for c in closes]
    df = pd.DataFrame(rows)
    df["time"] = pd.date_range("2025-01-01", periods=len(rows), freq="D", tz="UTC")
    return df


def test_replay_one_breakout_then_stop():
    trades, eq = replay_history("BTC", _ich([90, 95, 110, 112, 103]), RiskManager(), 0.6, log)
    assert len(trades) == 1 and trades[0].exit_reason == "stop_loss" and trades[0].bars_held == 2


def test_replay_open_at_end_is_force_closed():
    trades, eq = replay_history("BTC", _ich([90, 95, 110, 112, 120]), RiskManager(), 0.6, log)
    assert len(trades) == 1 and trades[0].exit_reason == "end_of_backtest"


def test_replay_no_signal_no_trades():
    trades, eq = replay_history("BTC", _ich([90, 91, 92, 93, 94]), RiskManager(), 0.6, log)
    assert trades == [] and eq == [1000.0]


def _cfg(markets):
    return types.SimpleNamespace(
        trading=types.SimpleNamespace(markets=markets, only_completed_candles=True, max_leverage=1.0),
        ichimoku=types.SimpleNamespace(conversion_periods=20, base_periods=60, span_b_periods=120, displacement=30),
        risk=types.SimpleNamespace(account_equity_usd=1000.0, max_capital_per_trade_frac=0.10,
                                   max_portfolio_exposure_frac=0.50, stop_loss_frac=0.05, take_profit_frac=0.15,
                                   use_trailing_stop=False, trailing_stop_frac=0.07, min_signal_confidence=0.6))


class FakeData:
    def __init__(self, df):
        self.df = df

    def fetch_daily(self, coin, lookback_days=700, drop_incomplete=True):
        return self.df.copy()


def test_backtester_skips_insufficient_history():
    short = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC"),
                          "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.0, "volume": 1.0})
    assert Backtester(_cfg(["BTC"]), FakeData(short), log, days=700).run()["BTC"]["metrics"]["trades"] == 0


def test_backtester_runs_and_reports_keys():
    import numpy as np
    n = 200
    rng = np.random.default_rng(1)
    base = 100 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
                       "open": base, "high": base + 3, "low": base - 3, "close": base, "volume": np.ones(n)})
    m = Backtester(_cfg(["BTC"]), FakeData(df), log, days=700).run()["BTC"]["metrics"]
    for k in ("trades", "win_rate", "total_return_pct", "profit_factor", "max_drawdown_pct"):
        assert k in m


def test_signal_attribution_aggregates():
    trades = [Trade("BTC", "d1", 100, "d2", 110, 1, 10, 10, 2, "take_profit",
                    entry_signals=("price_breakout_above_cloud", "flat_kijun_bull")),
              Trade("BTC", "d3", 100, "d4", 95, 1, -5, -5, 3, "stop_loss",
                    entry_signals=("flat_kijun_bull",))]
    agg = signal_attribution(trades)
    assert agg["price_breakout_above_cloud"]["trades"] == 1 and agg["flat_kijun_bull"]["trades"] == 2
    assert agg["flat_kijun_bull"]["pnl"] == pytest.approx(5.0)


def test_replay_records_entry_signals():
    trades, eq = replay_history("BTC", _ich([90, 95, 110, 112, 103]), RiskManager(), 0.6, log)
    assert "price_breakout_above_cloud" in trades[0].entry_signals


# --- NEW: save-results feature --------------------------------------------

def test_run_filename_shape_and_no_colons():
    name = _run_filename(1500, "txt")
    assert name.startswith("backtest_") and name.endswith("_1500d.txt") and ":" not in name
    assert re.match(r"^backtest_\d{4}-\d{2}-\d{2}_\d{4}_1500d\.txt$", name)


def test_run_filename_deterministic_with_now():
    assert _run_filename(700, "json", now=datetime(2026, 6, 30, 14, 30)) == "backtest_2026-06-30_1430_700d.json"


def test_save_results_writes_both_files(tmp_path):
    trades = [Trade("BTC", "d1", 100, "d2", 110, 1, 10, 10, 2, "take_profit",
                    entry_signals=("price_breakout_above_cloud",))]
    results = {"BTC": {"trades": trades, "equity_curve": [1000.0, 1010.0],
                       "metrics": compute_metrics(trades, [1000.0, 1010.0], 1000.0)}}
    txt_path, json_path = save_results(results, 1000.0, 1500, out_dir=str(tmp_path))
    assert txt_path.exists() and json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["days"] == 1500 and "BTC" in payload["markets"]
    assert payload["markets"]["BTC"]["trades"][0]["pnl"] == 10