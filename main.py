"""Entry point.

- Milestone 1: load + validate config, init logging, print mode banner.
- Milestone 2: fetch completed daily candles for each configured market (read-only)
- Milestone 3: compute ichimoku cloud and log where price sits vs. cloud
- Milestone 4a: evaluate the five core signals on the latest candle + a history scan
- Milestone 4b: extend signals to include c-clamp, flat kijun, e2e, etc
- Milestone 5: turn entry recommendations into respectively sized, risk-checked decisions
- Milestone 6: run the dry-run executor. Open/track/close paper positions, persisted to
  data/positions.json
- Milestone 7: engine/scheduler loop. main module modified to be a thin launcher
- Milestone 8: live executor. --live + ENABLE_LIVE_TRADING arms real orders on Hyperliquid,
  behind reconciliation, a typed confirmation, and a hard per-order size cap.

Pass --loop to keep a single long-running process that wakes shortly after 00:00 UTC.
Pass --live (with ENABLE_LIVE_TRADING=true in .env) to trade for real; --testnet to use
the Hyperliquid testnet; --max-order-usd to set the hard per-order notional ceiling.

Default: DRY RUN. Places no real orders.
"""

from __future__ import annotations

import argparse
import sys

from ichibot.config import ConfigError, load_config
from ichibot.engine import Engine
from ichibot.executor_dryrun import DryRunExecutor, ExecutorError, PositionStore
from ichibot.executor_live import LiveExecutor, LiveTradingError
from ichibot.logging_setup import setup_logging
from ichibot.market_data import HyperliquidData, MarketDataError
from ichibot.risk import RiskManager


def _build_live_executor(cfg, risk, store, log, args):
    """Construct the live executor behind the arming guardrails. Returns the executor,
    or None if the user declines confirmation (caller should exit)."""
    from eth_account import Account
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    base_url = constants.TESTNET_API_URL if args.testnet else constants.MAINNET_API_URL
    network = "TESTNET" if args.testnet else "MAINNET (REAL MONEY)"

    wallet = Account.from_key(cfg.private_key)          # AGENT key signs   <-- rename if needed
    exchange = Exchange(wallet, base_url, account_address=cfg.account_address)   # MAIN address owns
    info = Info(base_url, skip_ws=True)
    sz_decimals = {u["name"]: u["szDecimals"] for u in info.meta()["universe"]}

    # Guardrail 4: explicit typed confirmation before the first live order.
    print("\n" + "=" * 60)
    print("  LIVE TRADING ARMING")
    print(f"  Network         : {network}")
    print(f"  Account         : {cfg.account_address}")
    print(f"  Max order size  : ${args.max_order_usd:.2f}")
    print(f"  Markets         : {', '.join(cfg.trading.markets)}")
    print("=" * 60)
    if input("Type YES (all caps) to arm live trading: ").strip() != "YES":
        print("Not confirmed. Exiting.")
        return None

    executor = LiveExecutor(risk, log, exchange, info, cfg.account_address,
                            sz_decimals, store, max_order_usd=args.max_order_usd)
    log.warning("[LIVE] Armed on %s, max_order_usd=$%.2f", network, args.max_order_usd)
    return executor


def main() -> int:
    parser = argparse.ArgumentParser(description="ichibot -- Ichimoku trading bot")
    parser.add_argument("--loop", action="store_true",
                        help="run continuously, once shortly after each 00:00 UTC")
    parser.add_argument("--live", action="store_true",
                        help="arm LIVE trading (also requires ENABLE_LIVE_TRADING=true in .env)")
    parser.add_argument("--testnet", action="store_true",
                        help="use the Hyperliquid testnet instead of mainnet")
    parser.add_argument("--max-order-usd", type=float, default=25.0,
                        help="hard per-order notional ceiling (default $25)")
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("ichibot starting up")

    env_file = ".env.testnet" if args.testnet else ".env"
    try:
        cfg = load_config("Config.yaml", env_file)
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        log.error("Fix Config.yaml / .env and try again. Exiting.")
        return 1

    # --- decide mode: live requires BOTH the env master switch AND the --live flag ---
    want_live = args.live and cfg.enable_live_trading
    if args.live and not cfg.enable_live_trading:
        log.error("Refusing to go live: --live passed but ENABLE_LIVE_TRADING is not true in .env.")
        log.error("Running DRY RUN instead.")
    if cfg.enable_live_trading and not args.live:
        log.info("ENABLE_LIVE_TRADING is true, but --live was not passed. Running DRY RUN.")

    if want_live:
        log.warning("!" * 60)
        log.warning("LIVE TRADING ARMED. Real orders may be placed.")
        log.warning("!" * 60)
    else:
        log.info("Mode: DRY RUN. No real orders will ever be placed.")

    if want_live and args.testnet: 
        log.warning("[LIVE] Testnet execution, but candle data from mainnet.")

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
        if want_live:
            executor = _build_live_executor(cfg, risk, store, log, args)
            if executor is None:
                return 0        # user declined the confirmation prompt
        else:
            executor = DryRunExecutor(risk, log, store=store)
    except ExecutorError as exc:
        log.error("Could not load positions: %s", exc)
        return 1
    log.info("Loaded %d open position(s).", len(executor.positions))

    try:
        data = HyperliquidData()
    except MarketDataError as exc:
        log.error("Market data unavailable: %s", exc)
        log.error("Check your internet connection and try again. Exiting.")
        return 1

    engine = Engine(cfg, data, executor, log)

    try:
        if args.loop:
            try:
                engine.run_forever()
            except KeyboardInterrupt:
                log.info("Scheduler loop stopped by user.")
        else:
            engine.run_once()
            log.info("Book: %d open | exposure $%.2f / $%.2f | session realized PnL $%.2f",
                     len(executor.positions), executor.current_exposure_usd(),
                     risk.max_exposure_usd, executor.realized_pnl)
    except LiveTradingError as exc:
        log.error("[LIVE] HALTED: %s", exc)
        print(f"\n!!! LIVE TRADING HALTED: {exc}")
        print("!!! Resolve the divergence manually before restarting.")
        return 1

    log.info("Milestone 8 OK.")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())