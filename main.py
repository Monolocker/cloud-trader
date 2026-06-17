"""Entry point. 

Milestone 1: load + validate config, init logging, print mode banner.
Milestone 2: fetch completed candle for each configured market (read-only)

No trading logic exists yet. Will arrive in later milestones. Places no orders atm  
"""

from __future__ import annotations

import sys

from ichibot.config import ConfigError, load_config
from ichibot.logging_setup import setup_logging
from ichibot.market_data import HyperliquidData, MarketDataError


def main() -> int:
    log = setup_logging()
    log.info("=" * 60)
    log.info("ichibot starting up")

    try:
        cfg = load_config("Config.yaml", ".env")
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        log.error("Fix Config.yaml / .env and try again. Exiting.")
        return 1

    if cfg.enable_live_trading:
        log.warning("!" * 60)
        log.warning("LIVE TRADING IS ENABLED. Real orders could be placed.")
        log.warning("(No trading logic exists yet — nothing will trade.)")
        log.warning("!" * 60)
    else:
        log.info("Mode: DRY RUN. No real orders will ever be placed.")

    log.info("Config summary: %s", cfg.summary())

    # Milestone 2: read-only market data 

    try:
        data = HyperliquidData()
    except MarketDataError as exc:
        log.error("Market data unavailable: %s", exc)
        log.error("Check your internet connection and try again. Exiting.")
        return 1

    log.info("Fetching completed daily candles for %d market(s)...", len(cfg.trading.markets))
    fetched = 0
    for coin in cfg.trading.markets:
        try:
            df = data.fetch_daily(
                coin,
                lookback_days=200,
                drop_incomplete=cfg.trading.only_completed_candles,
            )
        except MarketDataError as exc:
            log.warning("Skipping %s: %s", coin, exc)
            continue

        if df.empty:
            log.warning("%s: no candle data returned", coin)
            continue

        last = df.iloc[-1]
        log.info(
            "%-6s | %4d candles | latest completed %s | close=%.4f",
            coin, len(df), last["time"].date(), last["close"],
        )
        fetched += 1

    log.info("Milestone 2 OK: fetched data for %d/%d markets.", fetched, len(cfg.trading.markets))
    log.info("=" * 60)
    return 0 if fetched > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

