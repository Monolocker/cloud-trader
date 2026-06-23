"""Entry point. 

- Milestone 1: load + validate config, init logging, print mode banner.
- ilestone 2: fetch completed daily candles for each configured market (read-only)
- Milestone 3: compute ichimoku cloud and log where price sits vs. cloud
- Milestone 4a: evaluate the five core signals on the latest candle + a history scan
- Milestone 4b: extend signals to include c-clamp, flat kijun, e2e, etc
- Milestone 5: turn entry recommendations into respectively sized, risk-checked decisions 
- Milestone 6: run the dry-run executor. Open/track/close paper positions, persisted to
data/positions.json
- Milestone 7: engine/scheduler loop. main module modified to be a thin launcher

Pass --loop to keep a single long-running process that wakes shortly after 00:00 UTC
              

Dry run: places no orders atm
"""

from __future__ import annotations

import argparse
import sys

from ichibot.config import ConfigError, load_config
from ichibot.engine import Engine
from ichibot.executor_dryrun import DryRunExecutor, ExecutorError, PositionStore
from ichibot.logging_setup import setup_logging
from ichibot.market_data import HyperliquidData, MarketDataError
from ichibot.risk import RiskManager



def main() -> int:
    parser = argparse.ArgumentParser(description="ichibot -- Ichimoku dry-run trading bot")
    parser.add_argument("--loop", action="store_true",
                        help="run continuously, once shortly after each 00:00 UTC")
    args = parser.parse_args()

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
        log.warning("LIVE TRADING IS ENABLED in .env.")
        log.warning("(No live executor is wired in yet -- nothing will trade live.)")
        log.warning("!" * 60)
    else:
        log.info("Mode: DRY RUN. No real orders will ever be placed.")

    log.info("Config summary: %s", cfg.summary())

    risk = RiskManager.from_config(cfg.risk, cfg.trading.max_leverage)
    log.info(
        "Risk: equity=$%.2f | per-trade cap=$%.2f | max exposure=$%.2f | "
        "stop=-%.1f%% tp=+%.1f%% | min_conf=%.2f",
        risk.account_equity_usd, risk.per_trade_cap_usd, risk.max_exposure_usd,
        cfg.risk.stop_loss_frac * 100, cfg.risk.take_profit_frac * 100, risk.min_signal_confidence,
    )

    try:
        store = PositionStore("data/positions.json")
        executor = DryRunExecutor(risk, log, store=store)
    except ExecutorError as exc:
        log.error("Could not load paper positions: %s", exc)
        return 1
    log.info("Loaded %d open paper position(s).", len(executor.positions))

    try:
        data = HyperliquidData()
    except MarketDataError as exc:
        log.error("Market data unavailable: %s", exc)
        log.error("Check your internet connection and try again. Exiting.")
        return 1

    engine = Engine(cfg, data, executor, log)

    if args.loop:
        try:
            engine.run_forever()
        except KeyboardInterrupt:
            log.info("Scheduler loop stopped by user.")
    else:
        engine.run_once()
        log.info("Paper book: %d open | exposure $%.2f / $%.2f | session realized PnL $%.2f",
                 len(executor.positions), executor.current_exposure_usd(),
                 risk.max_exposure_usd, executor.realized_pnl)

    log.info("Milestone 7 OK.")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

