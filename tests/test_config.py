"""Tests for config loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ichibot.config import ConfigError, load_config

VALID_YAML = """
trading:
  markets: ["BTC", "ETH", "SOL", "HYPE"]
  timeframe: "1d"
  max_leverage: 1
  only_completed_candles: true
ichimoku:
  conversion_periods: 20
  base_periods: 60
  span_b_periods: 120
  displacement: 30
risk:
  account_equity_usd: 1000.0
  max_capital_per_trade_frac: 0.10
  max_portfolio_exposure_frac: 0.50
  stop_loss_frac: 0.05
  take_profit_frac: 0.15
  use_trailing_stop: false
  trailing_stop_frac: 0.07
  min_signal_confidence: 0.6
"""


def _write(tmp_path: Path, body: str) -> str:
    p = tmp_path / "Config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


def test_valid_config_loads(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    cfg = load_config(_write(tmp_path, VALID_YAML), env_path=None)
    assert cfg.enable_live_trading is False
    assert cfg.trading.markets == ["BTC", "ETH", "SOL", "HYPE"]
    assert cfg.trading.max_leverage == 1.0
    assert "has_private_key" in cfg.summary()


def test_leverage_above_one_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    bad = VALID_YAML.replace("max_leverage: 1", "max_leverage: 3")
    with pytest.raises(ConfigError, match="max_leverage"):
        load_config(_write(tmp_path, bad), env_path=None)


def test_bad_confidence_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    bad = VALID_YAML.replace("min_signal_confidence: 0.6", "min_signal_confidence: 1.5")
    with pytest.raises(ConfigError, match="min_signal_confidence"):
        load_config(_write(tmp_path, bad), env_path=None)


def test_empty_markets_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    bad = VALID_YAML.replace('markets: ["BTC", "ETH", "SOL", "HYPE"]', "markets: []")
    with pytest.raises(ConfigError, match="markets"):
        load_config(_write(tmp_path, bad), env_path=None)


def test_per_trade_cannot_exceed_portfolio(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    bad = VALID_YAML.replace("max_capital_per_trade_frac: 0.10", "max_capital_per_trade_frac: 0.80")
    with pytest.raises(ConfigError, match="exceed"):
        load_config(_write(tmp_path, bad), env_path=None)


def test_live_trading_requires_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.delenv("HYPERLIQUID_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("HYPERLIQUID_ACCOUNT_ADDRESS", raising=False)
    with pytest.raises(ConfigError, match="HYPERLIQUID_PRIVATE_KEY"):
        load_config(_write(tmp_path, VALID_YAML), env_path=None)
