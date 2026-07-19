"""Tests for the backtest module. Network-free."""

from __future__ import annotations
import json
import logging, re, types
from datetime import datetime
import pandas as pd
import pytest
from ichibot.backtest import (Backtester, Trade, compute_metrics, max_drawdown,
                              replay_history, signal_attribution, _run_filename, save_results, buy_and_hold_return_pct)
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
    assert trades == []
    # Marked-to-market every bar: a flat account is a constant equity line,
    # one point per candle plus the starting point.
    assert eq == [1000.0] * len(_ich([90, 91, 92, 93, 94]))


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
    result = Backtester(_cfg(["BTC"]), FakeData(df), log, days=700).run()["BTC"]
    assert "buy_hold_pct" in result
    m = result["metrics"]
    for k in ("trades", "win_rate", "total_return_pct", "profit_factor", "max_drawdown_pct"):
        assert k in m

def test_buy_and_hold_return():
    # first valid close 100 -> last 150 == +50%; warmup NaN rows are excluded
    ich = _ich([90, 95, 110, 112, 150])
    assert buy_and_hold_return_pct(ich) == pytest.approx((150 / 90 - 1) * 100)

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


# --- save-results feature --------------------------------------------

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
                       "buy_hold_pct": 12.5,
                       "metrics": compute_metrics(trades, [1000.0, 1010.0], 1000.0)}}
    txt_path, json_path = save_results(results, 1000.0, 1500, out_dir=str(tmp_path))
    assert txt_path.exists() and json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["days"] == 1500 and "BTC" in payload["markets"]
    assert payload["markets"]["BTC"]["trades"][0]["pnl"] == 10

# Regression: the O(n) precomputed-flags replay must reproduce the EXACT trade
# list of the original O(n^2) grow-the-window replay. The reference below is
# the pre-refactor replay_history body, kept verbatim.

import numpy as np

from ichibot.backtest import Trade, replay_history
from ichibot.executor_dryrun import DryRunExecutor
from ichibot.ichimoku import compute_ichimoku
from ichibot.risk import RiskManager
from ichibot.signals import evaluate_signals


def _reference_replay_incremental(coin, ich, risk, min_confidence, logger):
    """Pre-refactor replay_history, verbatim (grow-the-window rescan)."""
    ex = DryRunExecutor(risk, logger, store=None); start = risk.account_equity_usd
    trades = []; eq = [start]; realized = 0.0; oi = {}

    def d(row, i):
        return str(row["time"].date()) if "time" in ich.columns else str(i)

    for i in range(1, len(ich)):
        w = ich.iloc[: i + 1]; row = w.iloc[-1]; price = float(row["close"])
        sig = evaluate_signals(w, min_confidence)
        action = ex.process(coin, price, sig)
        if action == "opened":
            pos = ex.positions[coin]
            oi = {"date": d(row, i), "price": pos.entry_price, "size": pos.size_units,
                  "i": i, "signals": tuple(sig.bullish_signals)}
        elif action.startswith("closed:"):
            reason = action.split(":", 1)[1]
            pnl = (price - oi["price"]) * oi["size"]; pct = (price / oi["price"] - 1) * 100
            realized += pnl
            trades.append(Trade(coin, oi["date"], oi["price"], d(row, i), price, oi["size"],
                                pnl, pct, i - oi["i"], reason, entry_signals=oi["signals"]))
            eq.append(start + realized)

    if coin in ex.positions:
        last = ich.iloc[-1]; price = float(last["close"]); ex.close_position(coin, price, "end_of_backtest")
        pnl = (price - oi["price"]) * oi["size"]; pct = (price / oi["price"] - 1) * 100; realized += pnl
        trades.append(Trade(coin, oi["date"], oi["price"], d(last, len(ich) - 1), price, oi["size"],
                            pnl, pct, (len(ich) - 1) - oi["i"], "end_of_backtest", entry_signals=oi["signals"]))
        eq.append(start + realized)
    return trades, eq


def _synthetic_ich(n=600, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.025, n)))
    df = pd.DataFrame({
        "time": pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC"),
        "open": close,
        "high": close * (1 + rng.uniform(0, 0.012, n)),
        "low": close * (1 - rng.uniform(0, 0.012, n)),
        "close": close,
        "volume": np.ones(n),
    })
    return compute_ichimoku(df, conversion_periods=20, base_periods=60,
                            span_b_periods=120, displacement=30)


def test_replay_matches_incremental_reference():
    log = logging.getLogger("regression"); log.addHandler(logging.NullHandler())
    for seed in (42, 7, 2024):  # several independent price paths
        ich = _synthetic_ich(seed=seed)
        ref_trades, ref_eq = _reference_replay_incremental(
            "BTC", ich, RiskManager(account_equity_usd=1000.0), 0.6, log)
        new_trades, new_eq = replay_history(
            "BTC", ich, RiskManager(account_equity_usd=1000.0), 0.6, log)
        assert len(ref_trades) > 0, "fixture produced no trades; strengthen it"
        # The reference predates the gross_pnl/fees_paid fields, so compare the
        # original behavioral fields exactly, then assert the zero-fee
        # invariants on the new fields.
        _LEGACY_FIELDS = ("coin", "entry_date", "entry_price", "exit_date", "exit_price",
                          "size_units", "pnl", "pnl_pct", "bars_held", "exit_reason",
                          "entry_signals")
        assert len(new_trades) == len(ref_trades)
        for new_t, ref_t in zip(new_trades, ref_trades):
            for fld in _LEGACY_FIELDS:
                assert getattr(new_t, fld) == getattr(ref_t, fld), fld
            assert new_t.gross_pnl == new_t.pnl      # zero fees: gross == net
            assert new_t.fees_paid == 0.0
        # Equity contract evolved with mark-to-market: the new curve has one
        # point per bar; the legacy close-only curve must survive inside it as
        # an ordered subsequence (flat-after-close bars equal start+realized,
        # bit-exact), with identical endpoints.
        assert len(new_eq) == len(ich)
        assert new_eq[0] == ref_eq[0] and new_eq[-1] == ref_eq[-1]
        it = iter(new_eq)
        assert all(any(v == x for x in it) for v in ref_eq), \
            "legacy equity points missing or out of order in MTM curve"


# Fees: hand-calculated math + zero-fee equivalence
from ichibot.config import FeesConfig

def test_fee_math_hand_calculated():
    """Entry 110, stop-exit 103, taker 0.1% both sides. Every number below is
    derivable by hand from the fixture and the 10%-of-$1000 sizing cap."""
    fees = FeesConfig(taker_fee_rate=0.001, maker_fee_rate=0.0005)
    trades, eq = replay_history("BTC", _ich([90, 95, 110, 112, 103]),
                                RiskManager(), 0.6, log, fees=fees)
    assert len(trades) == 1
    t = trades[0]
    size = t.size_units                       # 100 USD cap / 110 = 0.909090...
    assert size == pytest.approx(100.0 / 110.0)
    assert t.gross_pnl == pytest.approx((103.0 - 110.0) * size)         # -6.3636
    expected_fees = 110.0 * size * 0.001 + 103.0 * size * 0.001         #  0.19364
    assert t.fees_paid == pytest.approx(expected_fees)
    assert t.pnl == pytest.approx(t.gross_pnl - expected_fees)
    assert t.pnl_pct == pytest.approx(
        (103.0 / 110.0 - 1) * 100 - expected_fees / (110.0 * size) * 100)
    assert eq[-1] == pytest.approx(1000.0 + t.pnl)
 
 
def test_maker_rates_used_when_flagged():
    fees = FeesConfig(taker_fee_rate=0.001, maker_fee_rate=0.0002,
                      entry_is_taker=False, exit_is_taker=False)
    trades, _ = replay_history("BTC", _ich([90, 95, 110, 112, 103]),
                               RiskManager(), 0.6, log, fees=fees)
    t = trades[0]
    assert t.fees_paid == pytest.approx((110.0 + 103.0) * t.size_units * 0.0002)
 
 
def test_zero_rate_feesconfig_is_bit_identical_to_none():
    ich = _synthetic_ich(seed=42)
    a, ea = replay_history("BTC", ich, RiskManager(account_equity_usd=1000.0), 0.6, log,
                           fees=FeesConfig())
    b, eb = replay_history("BTC", ich, RiskManager(account_equity_usd=1000.0), 0.6, log,
                           fees=None)
    assert a == b and ea == eb   # full dataclass equality, every field
 
 
def test_fees_reduce_final_equity():
    ich = _synthetic_ich(seed=42)
    _, eq_free = replay_history("BTC", ich, RiskManager(account_equity_usd=1000.0), 0.6, log)
    trades, eq_fee = replay_history("BTC", ich, RiskManager(account_equity_usd=1000.0), 0.6, log,
                                    fees=FeesConfig(taker_fee_rate=0.00045))
    assert eq_fee[-1] < eq_free[-1]
    assert eq_free[-1] - eq_fee[-1] == pytest.approx(sum(t.fees_paid for t in trades))


# Mark-to-market equity: hand-calculated curve + drawdown honesty 

def test_equity_marked_on_every_bar_hand_calculated():
    """Fixture closes [90, 95, 110, 112, 103]; entry at 110 (bar 2), stop at
    103 (bar 4). Every equity point below is derivable by hand."""
    trades, eq = replay_history("BTC", _ich([90, 95, 110, 112, 103]),
                                RiskManager(), 0.6, log)
    s = trades[0].size_units                       # 100/110 units
    assert len(eq) == 5                            # start + one point per processed bar
    assert eq[0] == 1000.0                         # start
    assert eq[1] == 1000.0                         # bar 1: flat
    assert eq[2] == pytest.approx(1000.0)          # bar 2: entered AT the close -> 0 unrealized
    assert eq[3] == pytest.approx(1000.0 + (112 - 110) * s)   # bar 3: riding, +$1.818
    assert eq[4] == pytest.approx(1000.0 + (103 - 110) * s)   # bar 4: stopped, realized
 
 
def test_entry_fee_hits_equity_at_entry_bar():
    fees = FeesConfig(taker_fee_rate=0.001)
    trades, eq = replay_history("BTC", _ich([90, 95, 110, 112, 103]),
                                RiskManager(), 0.6, log, fees=fees)
    s = trades[0].size_units
    entry_fee = 110.0 * s * 0.001
    assert eq[2] == pytest.approx(1000.0 - entry_fee)              # fee paid when paid
    assert eq[3] == pytest.approx(1000.0 + (112 - 110) * s - entry_fee)
    assert eq[4] == pytest.approx(1000.0 + trades[0].pnl)          # both fees realized
 
 
def test_mtm_drawdown_at_least_legacy_drawdown():
    """The close-only curve hid intra-trade pain; the MTM curve may only ever
    reveal MORE drawdown, never less."""
    from ichibot.backtest import max_drawdown
    for seed in (42, 7, 2024):
        ich = _synthetic_ich(seed=seed)
        ref_trades, ref_eq = _reference_replay_incremental(
            "BTC", ich, RiskManager(account_equity_usd=1000.0), 0.6, log)
        _, new_eq = replay_history(
            "BTC", ich, RiskManager(account_equity_usd=1000.0), 0.6, log)
        assert max_drawdown(new_eq) >= max_drawdown(ref_eq) - 1e-12
 

# Funding (assumed constant rate): hand-calculated accrual and sign behavior.

def test_funding_hand_calculated():
    """Entry at 110 close of bar 2; held through bars 3 and 4. Rate 0.01%/8h,
    3 periods per daily bar -> 0.03% of notional per held bar."""
    fees = FeesConfig(funding_rate_8h=0.0001, funding_periods_per_bar=3.0)
    trades, eq = replay_history("BTC", _ich([90, 95, 110, 112, 103]),
                                RiskManager(), 0.6, log, fees=fees)
    t = trades[0]; s = t.size_units
    expected = 0.0003 * (112.0 * s) + 0.0003 * (103.0 * s)   # bars 3 and 4
    assert t.funding_paid == pytest.approx(expected)
    assert t.fees_paid == 0.0
    assert t.pnl == pytest.approx(t.gross_pnl - expected)
    # equity path: bar 3 carries one bar of accrual, bar 4 both (realized)
    assert eq[3] == pytest.approx(1000.0 + (112 - 110) * s - 0.0003 * 112.0 * s)
    assert eq[4] == pytest.approx(1000.0 + t.pnl)
 
 
def test_no_funding_on_entry_bar():
    """A position opened at bar 2's close held zero hours during bar 2."""
    fees = FeesConfig(funding_rate_8h=0.0001)
    trades, eq = replay_history("BTC", _ich([90, 95, 110, 112, 103]),
                                RiskManager(), 0.6, log, fees=fees)
    assert eq[2] == pytest.approx(1000.0)   # no entry fee here, and no funding yet
 
 
def test_negative_funding_pays_the_long():
    ich = _ich([90, 95, 110, 112, 103])
    pos, _ = replay_history("BTC", ich, RiskManager(), 0.6, log,
                            fees=FeesConfig(funding_rate_8h=0.0001))
    neg, _ = replay_history("BTC", ich, RiskManager(), 0.6, log,
                            fees=FeesConfig(funding_rate_8h=-0.0001))
    assert neg[0].funding_paid == pytest.approx(-pos[0].funding_paid)
    assert neg[0].pnl > pos[0].pnl