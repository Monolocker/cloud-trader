"""Entry point. 

Milestone 1: load + validate config, init logging, print mode banner.
Milestone 2: fetch completed daily candles for each configured market (read-only)
Milestone 3: compute ichimoku cloud and log where price sits vs. cloud

No trading logic exists yet. Will arrive in later milestones. Places no orders atm  
"""

from __future__ import annotations

import sys

from ichibot.config import ConfigError, load_config
from ichibot.ichimoku import cloud_position, compute_ichimoku, min_required_candles 
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
        log.warning("(No trading logic exists yet -- nothing will trade.)")
        log.warning("!" * 60)
    else:
        log.info("Mode: DRY RUN. No real orders will ever be placed.")

    log.info("Config summary: %s", cfg.summary())

    try:
        data = HyperliquidData()
    except MarketDataError as exc:
        log.error("Market data unavailable: %s", exc)
        log.error("Check your internet connection and try again. Exiting.")
        return 1

    # Milestone 3: Ichimoku computation

    needed = min_required_candles(cfg.ichimoku.span_b_periods, cfg.ichimoku.displacement)
    log.info("Scanning %d market(s); need >= %d candles per market for a full cloud.",
             len(cfg.trading.markets), needed)

    analyzed = 0
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

        if len(df) < needed:
            log.warning("%s: only %d candles, need %d -- skipping (market too new?)",
                        coin, len(df), needed)
            continue

        ich = compute_ichimoku(
            df,
            conversion_periods=cfg.ichimoku.conversion_periods,
            base_periods=cfg.ichimoku.base_periods,
            span_b_periods=cfg.ichimoku.span_b_periods,
            displacement=cfg.ichimoku.displacement,
        )
        last = ich.iloc[-1]
        log.info(
            "%-6s | %s close=%.4f | tenkan=%.4f kijun=%.4f | cloud=[%.4f, %.4f] | %s",
            coin, last["time"].date(), last["close"], last["tenkan"], last["kijun"],
            last["cloud_bottom"], last["cloud_top"], cloud_position(last),
        )
        analyzed += 1

    log.info("Milestone 3 OK: computed Ichimoku for %d/%d markets.",
             analyzed, len(cfg.trading.markets))
    log.info("=" * 60)
    return 0 if analyzed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())


