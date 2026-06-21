"""Tests for the dry-run executor and position store (Milestone 6). Network-free."""

from __future__ import annotations

import logging

import pytest

from ichibot.executor_dryrun import DryRunExecutor, ExecutorError, PositionStore
from ichibot.executor_live import LiveExecutor
from ichibot.risk import RiskManager
from ichibot.signals import SignalResult

log = logging.getLogger("ichibot.test")
BREAKOUT = ["price_breakout_above_cloud"]


def _sig(entry=False, exit=False, conf=0.0, bull=None, bear=None):
    return SignalResult(
        timestamp=None, bullish_signals=bull or [], bearish_signals=bear or [],
        confidence=conf, entry_recommended=entry, exit_recommended=exit, details={},
    )


def _entry_sig():
    return _sig(entry=True, conf=0.8, bull=BREAKOUT)


def test_opens_on_entry_signal():
    ex = DryRunExecutor(RiskManager(), log)
    assert ex.process("BTC", 100.0, _entry_sig()) == "opened"
    assert "BTC" in ex.positions
    assert ex.positions["BTC"].notional_usd == pytest.approx(100.0)


def test_no_entry_when_not_recommended():
    ex = DryRunExecutor(RiskManager(), log)
    assert ex.process("BTC", 100.0, _sig(entry=False)) == "none"
    assert "BTC" not in ex.positions


def test_no_duplicate_entry_while_holding():
    ex = DryRunExecutor(RiskManager(), log)
    ex.process("BTC", 100.0, _entry_sig())
    assert ex.process("BTC", 101.0, _entry_sig()) == "hold"
    assert len(ex.positions) == 1


def test_stop_loss_closes():
    ex = DryRunExecutor(RiskManager(), log)
    ex.process("BTC", 100.0, _entry_sig())
    assert ex.process("BTC", 94.0, _sig()) == "closed:stop_loss"
    assert "BTC" not in ex.positions


def test_take_profit_closes_and_records_pnl():
    ex = DryRunExecutor(RiskManager(), log)
    ex.process("BTC", 100.0, _entry_sig())
    assert ex.process("BTC", 116.0, _sig()) == "closed:take_profit"
    assert ex.realized_pnl == pytest.approx(16.0)


def test_signal_exit_closes():
    ex = DryRunExecutor(RiskManager(), log)
    ex.process("BTC", 100.0, _entry_sig())
    action = ex.process("BTC", 98.0, _sig(exit=True, bear=["price_below_tenkan"]))
    assert action == "closed:signal:price_below_tenkan"


def test_hold_when_in_position_and_quiet():
    ex = DryRunExecutor(RiskManager(), log)
    ex.process("BTC", 100.0, _entry_sig())
    assert ex.process("BTC", 105.0, _sig()) == "hold"


def test_trailing_stop_closes_after_peak():
    rm = RiskManager(use_trailing_stop=True, trailing_stop_frac=0.10, take_profit_frac=0.0)
    ex = DryRunExecutor(rm, log)
    ex.process("BTC", 100.0, _entry_sig())
    assert ex.process("BTC", 120.0, _sig()) == "hold"
    assert ex.positions["BTC"].peak_price == 120.0
    assert ex.process("BTC", 107.0, _sig()) == "closed:trailing_stop"


def test_exposure_cap_rejects_when_full():
    ex = DryRunExecutor(RiskManager(), log)
    for c in ["A", "B", "C", "D", "E"]:
        assert ex.process(c, 100.0, _entry_sig()) == "opened"
    assert ex.current_exposure_usd() == pytest.approx(500.0)
    assert ex.process("F", 100.0, _entry_sig()) == "entry_rejected"
    assert "F" not in ex.positions


def test_persistence_round_trip(tmp_path):
    store = PositionStore(str(tmp_path / "positions.json"))
    rm = RiskManager()
    ex = DryRunExecutor(rm, log, store=store)
    ex.process("BTC", 100.0, _entry_sig())
    ex.commit()
    ex2 = DryRunExecutor(rm, log, store=store)
    assert "BTC" in ex2.positions
    assert ex2.positions["BTC"].entry_price == 100.0
    assert ex2.positions["BTC"].notional_usd == pytest.approx(100.0)


def test_missing_file_loads_empty(tmp_path):
    store = PositionStore(str(tmp_path / "nope.json"))
    assert store.load() == {}


def test_corrupt_positions_file_raises(tmp_path):
    p = tmp_path / "positions.json"
    p.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ExecutorError):
        PositionStore(str(p)).load()


def test_live_executor_refuses_to_construct():
    with pytest.raises(NotImplementedError):
        LiveExecutor()