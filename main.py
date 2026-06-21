"""Entry point. 

- Milestone 1: load + validate config, init logging, print mode banner.
- ilestone 2: fetch completed daily candles for each configured market (read-only)
- Milestone 3: compute ichimoku cloud and log where price sits vs. cloud
- Milestone 4a: evaluate the five core signals on the latest candle + a history scan
- Milestone 4b: extend signals to include c-clamp, flat kijun, e2e, etc
- Milestone 5: turn entry recommendations into respectively sized, risk-checked decisions 
- Milestone 6: run the dry-run executor. Open/track/close paper positions, persisted to
data/positions.json
              

Entries/exits are only assessed, sized and logged only. Places no orders atm
"""

from __future__ import annotations

import sys

from ichibot.config import ConfigError, load_config
from ichibot.executor_dryrun import DryRunExecutor, ExecutorError, PositionStore
from ichibot.ichimoku import cloud_position, compute_ichimoku, min_required_candles 
from ichibot.logging_setup import setup_logging
from ichibot.market_data import HyperliquidData, MarketDataError
from ichibot.risk import RiskManager
from ichibot.signals import ALL_SIGNALS, evaluate_signals, signals_per_row



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
    log.info("Loaded %d open paper position(s) from %s", len(executor.positions), store.path)

    try:
        data = HyperliquidData()
    except MarketDataError as exc:
        log.error("Market data unavailable: %s", exc)
        log.error("Check your internet connection and try again. Exiting.")
        return 1

    needed = min_required_candles(cfg.ichimoku.span_b_periods, cfg.ichimoku.displacement)
    log.info("Scanning %d market(s); need >= %d candles per market.",
             len(cfg.trading.markets), needed)

    analyzed = 0
    for coin in cfg.trading.markets:
        try:
            df = data.fetch_daily(
                coin, lookback_days=200,
                drop_incomplete=cfg.trading.only_completed_candles,
            )
        except MarketDataError as exc:
            log.warning("Skipping %s: %s", coin, exc)
            continue

        if len(df) < needed:
            log.warning("%s: only %d candles, need %d -- skipping.", coin, len(df), needed)
            continue

        ich = compute_ichimoku(
            df,
            conversion_periods=cfg.ichimoku.conversion_periods,
            base_periods=cfg.ichimoku.base_periods,
            span_b_periods=cfg.ichimoku.span_b_periods,
            displacement=cfg.ichimoku.displacement,
        )
        last = ich.iloc[-1]
        price = float(last["close"])
        result = evaluate_signals(ich, cfg.risk.min_signal_confidence)
        held = "HELD" if coin in executor.positions else "flat"

        log.info("%-6s | %s close=%.4f | %s | %s | %s",
                 coin, last["time"].date(), price, cloud_position(last), held, result.summary())

        executor.process(coin, price, result)   # logs its own open/hold/close lines
        analyzed += 1

    executor.commit()
    log.info("Paper book: %d open | exposure $%.2f / $%.2f | session realized PnL $%.2f",
             len(executor.positions), executor.current_exposure_usd(),
             risk.max_exposure_usd, executor.realized_pnl)
    log.info("Milestone 6 OK: dry-run executor ran for %d/%d markets.",
             analyzed, len(cfg.trading.markets))
    log.info("=" * 60)
    return 0 if analyzed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

