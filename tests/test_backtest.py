"""Tests for the backtest module. Network-free."""

from __future__ import annotations

import logging
import types

import pandas as pd
import pytest

from ichibot.backtest import Backtester, Trade, compute_metrics, max_drawdown, replay_history
from ichibot.risk import RiskManager

log = logging.getLogger("ichibot.test")


def test_max_drawdown_basic():
    assert max_drawdown([1000, 1100, 900, 1000]) == pytest.approx(200 / 1100)


def test_max_drawdown_monotonic_is_zero():
    assert max_drawdown([1000, 1010, 1050, 1200]) == 0.0


def _trade(pnl, pnl_pct, bars=3):
    return Trade("X", "d1", 100.0, "d2", 100.0 + pnl, 1.0, pnl, pnl_pct, bars, "stop_loss")


def test_compute_metrics_basic():
    trades = [_trade(10, 10.0), _trade(-5, -5.0), _trade(20, 20.0)]
    eq = [1000.0, 1010.0, 1005.0, 1025.0]
    m = compute_metrics(trades, eq, 1000.0)
    assert m["trades"] == 3
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["total_return_pct"] == pytest.approx(2.5)
    assert m["profit_factor"] == pytest.approx(30 / 5)
    assert m["avg_win_pct"] == pytest.approx(15.0)
    assert m["avg_loss_pct"] == pytest.approx(-5.0)


def test_compute_metrics_empty():
    m = compute_metrics([], [1000.0], 1000.0)
    assert m["trades"] == 0
    assert m["win_rate"] == 0.0
    assert m["total_return_pct"] == 0.0
    assert m["profit_factor"] == 0.0


def _ich(closes, tenkan=90.0, kijun=85.0, cloud_top=100.0, cloud_bottom=80.0, a_fut=95.0, b_fut=90.0):
    rows = [{"close": float(c), "high": float(c), "low": float(c), "tenkan": tenkan, "kijun": kijun, 
             "cloud_top": cloud_top, "cloud_bottom": cloud_bottom,
             "senkou_a_future": a_fut, "senkou_b_future": b_fut} for c in closes]
    df = pd.DataFrame(rows)
    df["time"] = pd.date_range("2025-01-01", periods=len(rows), freq="D", tz="UTC")
    return df


def test_replay_one_breakout_then_stop():
    ich = _ich([90, 95, 110, 112, 103])
    trades, eq = replay_history("BTC", ich, RiskManager(), 0.6, log)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.entry_price == pytest.approx(110.0)
    assert t.bars_held == 2
    assert t.pnl == pytest.approx((103.0 - 110.0) * (100.0 / 110.0))
    assert eq[-1] == pytest.approx(1000.0 + t.pnl)


def test_replay_open_at_end_is_force_closed():
    ich = _ich([90, 95, 110, 112, 120])
    trades, eq = replay_history("BTC", ich, RiskManager(), 0.6, log)
    assert len(trades) == 1
    assert trades[0].exit_reason == "end_of_backtest"
    assert trades[0].pnl == pytest.approx((120.0 - 110.0) * (100.0 / 110.0))


def test_replay_no_signal_no_trades():
    ich = _ich([90, 91, 92, 93, 94])
    trades, eq = replay_history("BTC", ich, RiskManager(), 0.6, log)
    assert trades == []
    assert eq == [1000.0]


def _cfg(markets):
    return types.SimpleNamespace(
        trading=types.SimpleNamespace(markets=markets, only_completed_candles=True, max_leverage=1.0),
        ichimoku=types.SimpleNamespace(conversion_periods=20, base_periods=60,
                                       span_b_periods=120, displacement=30),
        risk=types.SimpleNamespace(
            account_equity_usd=1000.0, max_capital_per_trade_frac=0.10,
            max_portfolio_exposure_frac=0.50, stop_loss_frac=0.05, take_profit_frac=0.15,
            use_trailing_stop=False, trailing_stop_frac=0.07, min_signal_confidence=0.6),
    )


class FakeData:
    def __init__(self, df):
        self.df = df

    def fetch_daily(self, coin, lookback_days=700, drop_incomplete=True):
        return self.df.copy()


def test_backtester_skips_insufficient_history():
    short = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC"),
        "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.0, "volume": 1.0,
    })
    bt = Backtester(_cfg(["BTC"]), FakeData(short), log, days=700)
    results = bt.run()
    assert results["BTC"]["metrics"]["trades"] == 0


def test_backtester_runs_and_reports_keys():
    import numpy as np
    n = 200
    rng = np.random.default_rng(1)
    base = 100 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
        "open": base, "high": base + 3, "low": base - 3, "close": base, "volume": np.ones(n),
    })
    bt = Backtester(_cfg(["BTC"]), FakeData(df), log, days=700)
    results = bt.run()
    m = results["BTC"]["metrics"]
    for key in ("trades", "win_rate", "total_return_pct", "profit_factor", "max_drawdown_pct"):
        assert key in m