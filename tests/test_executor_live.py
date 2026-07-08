"""Tests for the live executor: guardrails + reconciliation, against fake SDK objects.
No real network or orders. The real order path is exercised by the user on testnet."""

from __future__ import annotations
import logging
import pytest
from ichibot.executor_live import LiveExecutor, LiveTradingError
from ichibot.risk import RiskManager, Position
from ichibot.signals import SignalResult

log = logging.getLogger("ichibot.test")
SZD = {"BTC": 5, "ETH": 4}


def _sig(entry=False, exit=False, conf=0.0, bull=None, bear=None):
    return SignalResult(None, bull or [], bear or [], conf, entry, exit, {})


def _entry_sig():
    return _sig(entry=True, conf=0.8, bull=["price_breakout_above_cloud"])


class FakeExchange:
    """Records order calls; returns a canned fill unless told to fail."""
    def __init__(self, fill_px=100.0, fail=False):
        self.fill_px = fill_px; self.fail = fail; self.calls = []

    def _resp(self, sz):
        if self.fail:
            return {"status": "ok", "response": {"data": {"statuses": [{"error": "insufficient margin"}]}}}
        return {"status": "ok", "response": {"data": {"statuses": [
            {"filled": {"avgPx": str(self.fill_px), "totalSz": str(sz), "oid": 1}}]}}}

    def market_open(self, name, is_buy, sz, **kw):
        self.calls.append(("open", name, is_buy, sz)); return self._resp(sz)

    def market_close(self, coin, **kw):
        self.calls.append(("close", coin)); return self._resp(0.0)


class FakeInfo:
    def __init__(self, asset_positions=None):
        self._ap = asset_positions or []

    def user_state(self, address):
        return {"assetPositions": self._ap}


def _pos(coin="BTC", entry=100.0, size=0.1):
    return Position(coin, entry, size, entry * size, entry * 0.95, None, entry)


def _exec(exchange=None, info=None, positions=None, max_order_usd=25.0, equity=1000.0):
    ex = LiveExecutor(RiskManager(account_equity_usd=equity), log,
                      exchange or FakeExchange(), info or FakeInfo(),
                      "0xMAIN", SZD, store=None, max_order_usd=max_order_usd)
    if positions:
        ex.positions = positions
    return ex


def test_reconcile_ok_when_flat_and_exchange_flat():
    _exec(info=FakeInfo([])).reconcile()


def test_reconcile_ok_when_sizes_match():
    ex = _exec(info=FakeInfo([{"position": {"coin": "BTC", "szi": "0.1"}}]),
               positions={"BTC": _pos("BTC", size=0.1)})
    ex.reconcile()


def test_reconcile_halts_on_size_mismatch():
    ex = _exec(info=FakeInfo([{"position": {"coin": "BTC", "szi": "0.5"}}]),
               positions={"BTC": _pos("BTC", size=0.1)})
    with pytest.raises(LiveTradingError):
        ex.reconcile()


def test_reconcile_halts_when_exchange_has_unknown_position():
    ex = _exec(info=FakeInfo([{"position": {"coin": "ETH", "szi": "1.0"}}]), positions={})
    with pytest.raises(LiveTradingError):
        ex.reconcile()


def test_reconcile_halts_on_short_position():
    ex = _exec(info=FakeInfo([{"position": {"coin": "BTC", "szi": "-0.1"}}]),
               positions={"BTC": _pos("BTC", size=-0.1)})
    with pytest.raises(LiveTradingError):
        ex.reconcile()


def test_entry_places_long_market_order():
    fx = FakeExchange(fill_px=100.0)
    ex = _exec(exchange=fx)
    assert ex.process("BTC", 100.0, _entry_sig()) == "opened"
    assert fx.calls[0][:3] == ("open", "BTC", True)
    assert "BTC" in ex.positions


def test_entry_notional_clamped_by_max_order_usd():
    fx = FakeExchange(fill_px=100.0)
    ex = _exec(exchange=fx, max_order_usd=25.0)
    ex.process("BTC", 100.0, _entry_sig())
    _, _, _, sz = fx.calls[0]
    assert sz == pytest.approx(0.25, abs=1e-6)


def test_entry_size_rounded_to_szdecimals():
    fx = FakeExchange(fill_px=30000.0)
    ex = _exec(exchange=fx, max_order_usd=25.0)
    ex.process("BTC", 30000.0, _entry_sig())
    _, _, _, sz = fx.calls[0]
    assert sz == round(25.0 / 30000.0, 5)


def test_entry_skipped_when_capped_below_minimum():
    fx = FakeExchange()
    ex = _exec(exchange=fx, max_order_usd=5.0)
    assert ex.process("BTC", 100.0, _entry_sig()) == "entry_rejected"
    assert fx.calls == []


def test_entry_respects_1x_ceiling():
    fx = FakeExchange()
    ex = _exec(exchange=fx, max_order_usd=1000.0, equity=100.0)
    ex.positions = {"ETH": Position("ETH", 100.0, 1.0, 100.0, 95.0, None, 100.0)}
    assert ex.process("BTC", 100.0, _entry_sig()) == "entry_rejected"
    assert fx.calls == []


def test_entry_failure_does_not_record_position():
    fx = FakeExchange(fail=True)
    ex = _exec(exchange=fx)
    assert ex.process("BTC", 100.0, _entry_sig()) == "entry_failed"
    assert "BTC" not in ex.positions


def test_stop_loss_closes_via_market_close():
    fx = FakeExchange(fill_px=94.0)
    ex = _exec(exchange=fx, positions={"BTC": _pos("BTC", entry=100.0, size=0.1)})
    action = ex.process("BTC", 94.0, _sig())
    assert action == "closed:stop_loss"
    assert fx.calls[0] == ("close", "BTC")
    assert "BTC" not in ex.positions


def test_signal_exit_closes():
    fx = FakeExchange(fill_px=98.0)
    ex = _exec(exchange=fx, positions={"BTC": _pos("BTC", entry=100.0, size=0.1)})
    action = ex.process("BTC", 98.0, _sig(exit=True, bear=["price_below_kijun"]))
    assert action == "closed:signal:price_below_kijun"


def test_close_failure_keeps_position():
    fx = FakeExchange(fail=True)
    ex = _exec(exchange=fx, positions={"BTC": _pos("BTC", entry=100.0, size=0.1)})
    assert ex.process("BTC", 94.0, _sig()) == "close_failed"
    assert "BTC" in ex.positions