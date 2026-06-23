"""Tests for the orchestration engine + dedup (Milestone 7). Network-free."""

from __future__ import annotations

import logging
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ichibot.engine import Engine, seconds_until_next_utc_run

log = logging.getLogger("ichibot.test")


def _cfg(markets):
    return types.SimpleNamespace(
        trading=types.SimpleNamespace(markets=markets, only_completed_candles=True),
        ichimoku=types.SimpleNamespace(conversion_periods=20, base_periods=60,
                                       span_b_periods=120, displacement=30),
        risk=types.SimpleNamespace(min_signal_confidence=0.6),
    )


def _ohlc(n=160, seed=0, last_date="2026-06-19"):
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0, 2, n))
    times = pd.date_range(end=pd.Timestamp(last_date, tz="UTC"), periods=n, freq="D")
    return pd.DataFrame({
        "time": times, "open": base, "high": base + 2.0, "low": base - 2.0,
        "close": base, "volume": np.ones(n),
    })


class FakeData:
    def __init__(self, df):
        self.df = df

    def fetch_daily(self, coin, lookback_days=200, drop_incomplete=True):
        return self.df.copy()


class FakeExecutor:
    def __init__(self):
        self.calls = []
        self.commits = 0
        self.positions = {}

    def process(self, coin, price, signal):
        self.calls.append(coin)
        return "none"

    def commit(self):
        self.commits += 1


def test_run_once_processes_each_market(tmp_path):
    ex = FakeExecutor()
    eng = Engine(_cfg(["BTC", "ETH"]), FakeData(_ohlc()), ex, log, state_path=str(tmp_path / "s.json"))
    summary = eng.run_once()
    assert summary["processed"] == ["BTC", "ETH"]
    assert ex.calls == ["BTC", "ETH"]
    assert ex.commits == 1


def test_second_run_same_candle_skips(tmp_path):
    ex = FakeExecutor()
    eng = Engine(_cfg(["BTC", "ETH"]), FakeData(_ohlc()), ex, log, state_path=str(tmp_path / "s.json"))
    eng.run_once()
    summary = eng.run_once()
    assert summary["processed"] == []
    assert summary["skipped_dup"] == ["BTC", "ETH"]
    assert ex.calls == ["BTC", "ETH"]


def test_new_candle_processes_again(tmp_path):
    ex = FakeExecutor()
    data = FakeData(_ohlc(last_date="2026-06-19"))
    eng = Engine(_cfg(["BTC"]), data, ex, log, state_path=str(tmp_path / "s.json"))
    eng.run_once()
    data.df = _ohlc(last_date="2026-06-20")
    eng.run_once()
    assert ex.calls == ["BTC", "BTC"]


def test_state_persists_across_engine_instances(tmp_path):
    sp = str(tmp_path / "s.json")
    Engine(_cfg(["BTC"]), FakeData(_ohlc()), FakeExecutor(), log, state_path=sp).run_once()
    ex2 = FakeExecutor()
    summary = Engine(_cfg(["BTC"]), FakeData(_ohlc()), ex2, log, state_path=sp).run_once()
    assert summary["skipped_dup"] == ["BTC"]
    assert ex2.calls == []


def test_insufficient_candles_skipped(tmp_path):
    ex = FakeExecutor()
    eng = Engine(_cfg(["BTC"]), FakeData(_ohlc(n=100)), ex, log, state_path=str(tmp_path / "s.json"))
    summary = eng.run_once()
    assert summary["skipped_other"] == ["BTC"]
    assert ex.calls == []


def test_fetch_error_skips_market(tmp_path):
    class BadData:
        def fetch_daily(self, *a, **k):
            raise RuntimeError("boom")
    ex = FakeExecutor()
    eng = Engine(_cfg(["BTC"]), BadData(), ex, log, state_path=str(tmp_path / "s.json"))
    summary = eng.run_once()
    assert summary["skipped_other"] == ["BTC"]
    assert ex.calls == []


def test_corrupt_state_starts_fresh(tmp_path):
    sp = tmp_path / "s.json"
    sp.write_text("{ bad json", encoding="utf-8")
    ex = FakeExecutor()
    eng = Engine(_cfg(["BTC"]), FakeData(_ohlc()), ex, log, state_path=str(sp))
    summary = eng.run_once()
    assert summary["processed"] == ["BTC"]


def test_seconds_until_next_run():
    now = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    secs = seconds_until_next_utc_run(buffer_minutes=5, now=now)
    assert secs == pytest.approx(12 * 3600 + 5 * 60)
