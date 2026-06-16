"""Entry point. Milestone 1: load + validate config, init logging, print mode banner.

No market data and no trading logic exist yet. Will arrive in later milestones.
"""

from __future__ import annotations

import sys

from ichibot.config import ConfigError, load_config
from ichibot.logging_setup import setup_logging


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
        log.warning("(No trading logic exists yet at Milestone 1 — nothing will trade.)")
        log.warning("!" * 60)
    else:
        log.info("Mode: DRY RUN. No real orders will ever be placed.")

    log.info("Config summary: %s", cfg.summary())
    log.info("Milestone 1 OK: configuration validated and logging initialized.")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
