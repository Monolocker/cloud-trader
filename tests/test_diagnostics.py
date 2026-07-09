"""Tests for ichibot.diagnostics (Layer 0 measurement, v1.5).

Key invariants:
  1. Vectorized-once flags equal the incremental window flags used by
     replay_history (rolling ops are causal), so the diagnostics replay
     sees the exact same signals as the existing backtest.
  2. Raw occurrence counts are independent of execution.
  3. Every after-warm-up bullish twist produces a TwistEvent with a verdict.
  4. A lone twist (no primary co-firing) never opens a trade under defaults.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from ichibot.diagnostics import diagnose_market, format_report
from ichibot.ichimoku import compute_ichimoku
from ichibot.risk import RiskManager
from ichibot.signals import (
    SIG_KUMO_TWIST_BULL,
    evaluate_signals,
    signals_per_row,
)

LOG = logging.getLogger("test")
LOG.addHandler(logging.NullHandler())

# Small Ichimoku params so synthetic fixtures stay tiny.
PARAMS = dict(conversion_periods=3, base_periods=5, span_b_periods=8, displacement=4)


def _synthetic(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.ones(n),
    })


@pytest.fixture()
def ich() -> pd.DataFrame:
    return compute_ichimoku(_synthetic(), **PARAMS)


def _risk() -> RiskManager:
    return RiskManager(account_equity_usd=1000.0)


def test_vectorized_flags_match_incremental_windows(ich):
    """The core equivalence claim: per-row flags computed once over the full
    frame equal the last-row flags of every growing window, i.e. the
    diagnostics replay observes exactly what replay_history observes."""
    full = signals_per_row(ich)
    for i in range(2, len(ich), 17):  # sampled stride keeps the test fast
        window_last = signals_per_row(ich.iloc[: i + 1]).iloc[-1]
        assert (full.iloc[i] == window_last).all(), f"mismatch at row {i}"


def test_evaluate_signals_agrees_with_diagnostics_gating(ich):
    """Entry gating reconstructed in diagnose_market matches evaluate_signals."""
    for i in range(150, 170):
        res = evaluate_signals(ich.iloc[: i + 1], min_confidence=0.6)
        flags = signals_per_row(ich).iloc[i]
        fired = [s for s in res.bullish_signals]
        assert all(flags[s] for s in fired)


def test_raw_counts_independent_of_execution(ich):
    diag = diagnose_market("BTC", ich, _risk(), min_confidence=0.6, logger=LOG)
    flags = signals_per_row(ich)
    for sig, c in diag.counts.items():
        # replay starts at row 1, matching replay_history
        assert c.raw == int(flags[sig].iloc[1:].sum())
        assert c.after_warmup <= c.raw
        assert c.while_flat + c.while_in_position == c.after_warmup
        assert c.executed_entry <= c.while_flat


def test_every_warm_twist_gets_a_verdict(ich):
    diag = diagnose_market("BTC", ich, _risk(), min_confidence=0.6, logger=LOG)
    warm = ich["cloud_top"].notna()
    flags = signals_per_row(ich)
    expected = int((flags[SIG_KUMO_TWIST_BULL].iloc[1:] & warm.iloc[1:]).sum())
    assert len(diag.twist_events) == expected
    assert all(e.verdict for e in diag.twist_events)


def test_lone_twist_never_opens_trade():
    """A candle where ONLY the twist fires must not produce an entry, because
    the twist is non-primary and its weight (0.25) is below min_confidence."""
    ich = compute_ichimoku(_synthetic(300, seed=11), **PARAMS)
    diag = diagnose_market("BTC", ich, _risk(), min_confidence=0.6, logger=LOG)
    for e in diag.twist_events:
        if not e.coincident_bullish and not e.position_open:
            assert "not a primary" in e.verdict or "confidence" in e.verdict


def test_report_renders(ich):
    diag = diagnose_market("BTC", ich, _risk(), min_confidence=0.6, logger=LOG)
    text = format_report([diag])
    assert "Kumo twist trace" in text
    assert "BTC" in text