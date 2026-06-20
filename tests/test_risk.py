"""Tests for the risk manager (Milestone 5). Network-free, pure arithmetic."""

from __future__ import annotations

import pytest

from ichibot.risk import ExitDecision, Position, RiskError, RiskManager


def _rm(**over):
    base = dict(
        account_equity_usd=1000.0, max_capital_per_trade_frac=0.10,
        max_portfolio_exposure_frac=0.50, stop_loss_frac=0.05,
        take_profit_frac=0.15, use_trailing_stop=False, trailing_stop_frac=0.07,
        min_signal_confidence=0.6, max_leverage=1.0, min_order_usd=10.0,
    )
    base.update(over)
    return RiskManager(**base)


def test_basic_sizing_math():
    d = _rm().size_position("BTC", price=100.0, confidence=0.8)
    assert d.approved
    assert d.notional_usd == pytest.approx(100.0)
    assert d.size_units == pytest.approx(1.0)
    assert d.stop_price == pytest.approx(95.0)
    assert d.take_profit_price == pytest.approx(115.0)
    assert d.dollar_risk == pytest.approx(5.0)


def test_take_profit_disabled_when_zero():
    d = _rm(take_profit_frac=0.0).size_position("BTC", 100.0, 0.8)
    assert d.approved
    assert d.take_profit_price is None


def test_confidence_below_threshold_rejected():
    d = _rm().size_position("BTC", 100.0, confidence=0.59)
    assert not d.approved
    assert "confidence" in d.reason


def test_invalid_price_rejected():
    assert not _rm().size_position("BTC", 0.0, 0.8).approved
    assert not _rm().size_position("BTC", -5.0, 0.8).approved


def test_exposure_budget_clamps_notional():
    d = _rm().size_position("BTC", 100.0, 0.8, current_exposure_usd=450.0)
    assert d.approved
    assert d.notional_usd == pytest.approx(50.0)
    assert d.size_units == pytest.approx(0.5)


def test_exposure_full_rejected():
    d = _rm().size_position("BTC", 100.0, 0.8, current_exposure_usd=500.0)
    assert not d.approved
    assert "exposure" in d.reason


def test_remaining_below_min_order_rejected():
    d = _rm().size_position("BTC", 100.0, 0.8, current_exposure_usd=495.0)
    assert not d.approved
    assert "minimum order" in d.reason


def test_leverage_above_one_rejected_at_construction():
    with pytest.raises(RiskError):
        _rm(max_leverage=2.0)


def test_per_trade_above_exposure_rejected():
    with pytest.raises(RiskError):
        _rm(max_capital_per_trade_frac=0.8, max_portfolio_exposure_frac=0.5)


def test_bad_stop_rejected():
    with pytest.raises(RiskError):
        _rm(stop_loss_frac=0.0)


def _pos(**over):
    base = dict(coin="BTC", entry_price=100.0, size_units=1.0, notional_usd=100.0,
                stop_price=95.0, take_profit_price=115.0, peak_price=100.0)
    base.update(over)
    return Position(**base)


def test_stop_loss_triggers():
    d = _rm().evaluate_exit(_pos(), current_price=94.0)
    assert d.should_exit and d.reason == "stop_loss"


def test_take_profit_triggers():
    d = _rm().evaluate_exit(_pos(), current_price=116.0)
    assert d.should_exit and d.reason == "take_profit"


def test_no_exit_inside_band():
    d = _rm().evaluate_exit(_pos(), current_price=105.0)
    assert not d.should_exit and d.reason == ""


def test_trailing_stop_triggers_after_peak():
    rm = _rm(use_trailing_stop=True, trailing_stop_frac=0.07)
    pos = _pos(stop_price=80.0, take_profit_price=None, peak_price=100.0)
    rm.update_trailing_peak(pos, 120.0)
    assert pos.peak_price == 120.0
    d = rm.evaluate_exit(pos, current_price=111.0)
    assert d.should_exit and d.reason == "trailing_stop"


def test_trailing_not_triggered_above_trail():
    rm = _rm(use_trailing_stop=True, trailing_stop_frac=0.07)
    pos = _pos(stop_price=80.0, take_profit_price=None, peak_price=120.0)
    d = rm.evaluate_exit(pos, current_price=115.0)
    assert not d.should_exit


def test_position_from_decision():
    d = _rm().size_position("ETH", 50.0, 0.8)
    pos = Position.from_decision(d)
    assert pos.coin == "ETH" and pos.entry_price == 50.0
    assert pos.peak_price == 50.0


def test_position_from_rejected_decision_raises():
    d = _rm().size_position("ETH", 50.0, confidence=0.1)
    with pytest.raises(RiskError):
        Position.from_decision(d)