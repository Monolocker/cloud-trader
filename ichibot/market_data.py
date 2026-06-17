"""Fetch daily OHLCV candles from Hyperliquid and return them as a DataFrame.

Only reads public market data.
"""

from __future__ import annotations

import time

import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Every candle DataFrame is guaranteed to have these columns, in this order.
CANDLE_COLUMNS = ["time", "open", "high", "low", "close", "volume"]

MS_PER_DAY = 24 * 60 * 60 * 1000


class MarketDataError(Exception):
    """Raised when candle data cannot be fetched or is malformed."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def candles_to_dataframe(
    raw: list[dict],
    drop_incomplete: bool = True,
    now_ms: int | None = None,
) -> pd.DataFrame:
    """Convert Hyperliquid's raw candle list into a clean, typed DataFrame.

    Hyperliquid returns each candle as a dict like:
        {"t": <open ms>, "T": <close ms>, "o": "29295.0", "h": "...",
         "l": "...", "c": "...", "v": "...", "n": 189, "s": "BTC", "i": "1d"}
    OHLCV values arrive as STRINGS, so we cast them to float.

    The result is:
      - sorted by candle OPEN time, ascending,
      - given a 'time' column = candle open time as a pandas UTC Timestamp,
      - stripped of the still-forming candle (close time in the future) when
        drop_incomplete is True, so we only ever act on completed daily candles.
    """
    if now_ms is None:
        now_ms = _now_ms()

    if not isinstance(raw, list):
        raise MarketDataError(f"Expected a list of candles, got {type(raw).__name__}")

    rows = []
    for c in raw:
        try:
            rows.append(
                {
                    "open_ms": int(c["t"]),
                    "close_ms": int(c["T"]),
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(f"Malformed candle {c!r}: {exc}") from exc

    if not rows:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    df = pd.DataFrame(rows).sort_values("open_ms").reset_index(drop=True)

    if drop_incomplete:
        # A candle is complete only once its close time has passed.
        df = df[df["close_ms"] <= now_ms].reset_index(drop=True)

    df["time"] = pd.to_datetime(df["open_ms"], unit="ms", utc=True)
    return df[CANDLE_COLUMNS]


class HyperliquidData:
    """Thin, read-only wrapper around the Hyperliquid Info client."""

    def __init__(self, base_url: str = constants.MAINNET_API_URL):
        # Constructing Info makes network calls (it loads exchange metadata),
        # needs connectivity. skip_ws=True avoids opening a websocket.
        try:
            self._info = Info(base_url, skip_ws=True)
        except Exception as exc:  # network / SDK errors
            raise MarketDataError(f"Could not connect to Hyperliquid: {exc}") from exc

    def fetch_daily(
        self,
        coin: str,
        lookback_days: int = 200,
        drop_incomplete: bool = True,
    ) -> pd.DataFrame:
        """Fetch up to `lookback_days` of completed 1d candles for `coin`."""
        if lookback_days <= 0:
            raise MarketDataError("lookback_days must be positive")

        end_ms = _now_ms()
        
        # +2 days of slack so we still have enough completed candles after the incomplete-candle drop.
        start_ms = end_ms - (lookback_days + 2) * MS_PER_DAY

        try:
            raw = self._info.candles_snapshot(coin, "1d", start_ms, end_ms)
        except Exception as exc:
            raise MarketDataError(f"Failed to fetch candles for {coin}: {exc}") from exc

        return candles_to_dataframe(raw, drop_incomplete=drop_incomplete, now_ms=end_ms)