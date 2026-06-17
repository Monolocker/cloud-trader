"""Tests for candle parsing. Network-free, feeding sample candle
dicts (in Hyperliquid's documented shape) through the parser."""

from __future__ import annotations

import pytest

from ichibot.market_data import CANDLE_COLUMNS, MarketDataError, candles_to_dataframe

DAY = 24 * 60 * 60 * 1000


def _candle(t, o, h, l, c, v):
    """Build a raw candle dict matching Hyperliquid's format (strings for OHLCV)."""
    return {
        "t": t,
        "T": t + DAY - 1,
        "s": "BTC",
        "i": "1d",
        "o": str(o),
        "h": str(h),
        "l": str(l),
        "c": str(c),
        "v": str(v),
        "n": 10,
    }


def test_parses_and_casts_to_float():
    raw = [_candle(0, 100, 110, 90, 105, 1.5)]
    df = candles_to_dataframe(raw, drop_incomplete=True, now_ms=10 * DAY)
    assert list(df.columns) == CANDLE_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[0]["close"] == 105.0
    assert isinstance(df.iloc[0]["close"], float)


def test_sorted_by_open_time():
    raw = [_candle(2 * DAY, 1, 1, 1, 1, 1), _candle(0, 2, 2, 2, 2, 2), _candle(DAY, 3, 3, 3, 3, 3)]
    df = candles_to_dataframe(raw, drop_incomplete=False)
    times = df["time"].astype("int64").tolist()
    assert times == sorted(times)


def test_drops_incomplete_candle():
    raw = [_candle(0, 1, 1, 1, 1, 1), _candle(DAY, 2, 2, 2, 2, 2)]
    now = DAY + DAY // 2  # mid-way through the second candle => incomplete
    df = candles_to_dataframe(raw, drop_incomplete=True, now_ms=now)
    assert len(df) == 1
    assert df.iloc[0]["open"] == 1.0


def test_keeps_incomplete_when_flag_off():
    raw = [_candle(0, 1, 1, 1, 1, 1), _candle(DAY, 2, 2, 2, 2, 2)]
    now = DAY + DAY // 2
    df = candles_to_dataframe(raw, drop_incomplete=False, now_ms=now)
    assert len(df) == 2


def test_empty_returns_empty_frame_with_columns():
    df = candles_to_dataframe([], drop_incomplete=True)
    assert df.empty
    assert list(df.columns) == CANDLE_COLUMNS


def test_malformed_candle_raises():
    with pytest.raises(MarketDataError):
        candles_to_dataframe([{"t": 0}], drop_incomplete=False)
