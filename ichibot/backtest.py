"""Backtest the Ichimoku strategy over a long history (Backtest milestone).

Replays the exact current live pipeline: compute_ichimoku -> evaluate_signals ->
RiskManager -> DryRunExecutor -- candle-by-candle over a long lookback, and
reports per-market metrics (trades, win rate, return, profit factor, drawdown).

Each market is isolated upon simulation, as if it had the full account. This
answers "does the strategy have edge on this instrument?". A single shared-
portfolio simulation (cross-market exposure on one account) can be a later
enhancement.

This is a measurement tool. It places no orders and is not a promise of future
results -- past performance does not predict future performance.

Run:  python -m ichibot.backtest --days 700
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ichibot.executor_dryrun import DryRunExecutor
from ichibot.ichimoku import compute_ichimoku, min_required_candles
from ichibot.risk import RiskManager
from ichibot.signals import evaluate_signals


@dataclass
class Trade:
    coin: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    size_units: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str


def max_drawdown(equity_curve: list[float]) -> float:
    """Largest peak-to-trough decline on the (realized) equity curve, as a fraction."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def compute_metrics(trades: list[Trade], equity_curve: list[float], start_equity: float) -> dict:
    n = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)          # positive number
    final_equity = equity_curve[-1] if equity_curve else start_equity

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "trades": n,
        "win_rate": (len(wins) / n) if n else 0.0,
        "total_return_pct": (final_equity / start_equity - 1.0) * 100.0,
        "avg_win_pct": (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0.0,
        "avg_loss_pct": (sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown(equity_curve) * 100.0,
        "avg_bars_held": (sum(t.bars_held for t in trades) / n) if n else 0.0,
        "final_equity": final_equity,
    }


def replay_history(coin, ich, risk: RiskManager, min_confidence: float, logger) -> tuple[list[Trade], list[float]]:
    """Replay one market's Ichimoku DataFrame through the executor; return
    (completed trades, realized equity curve)."""
    ex = DryRunExecutor(risk, logger, store=None)
    start_equity = risk.account_equity_usd
    trades: list[Trade] = []
    equity_curve: list[float] = [start_equity]
    realized = 0.0
    open_info: dict = {}

    def _date(row, i):
        return str(row["time"].date()) if "time" in ich.columns else str(i)

    for i in range(1, len(ich)):
        window = ich.iloc[: i + 1]
        row = window.iloc[-1]
        price = float(row["close"])
        signal = evaluate_signals(window, min_confidence)
        action = ex.process(coin, price, signal)

        if action == "opened":
            pos = ex.positions[coin]
            open_info = {"date": _date(row, i), "price": pos.entry_price,
                         "size": pos.size_units, "i": i}
        elif action.startswith("closed:"):
            reason = action.split(":", 1)[1]
            pnl = (price - open_info["price"]) * open_info["size"]
            pnl_pct = (price / open_info["price"] - 1.0) * 100.0
            realized += pnl
            trades.append(Trade(coin, open_info["date"], open_info["price"], _date(row, i),
                                price, open_info["size"], pnl, pnl_pct, i - open_info["i"], reason))
            equity_curve.append(start_equity + realized)

    # Close any still-open position at the final candle, for a clean tally.
    if coin in ex.positions:
        last = ich.iloc[-1]
        price = float(last["close"])
        ex.close_position(coin, price, "end_of_backtest")
        pnl = (price - open_info["price"]) * open_info["size"]
        pnl_pct = (price / open_info["price"] - 1.0) * 100.0
        realized += pnl
        trades.append(Trade(coin, open_info["date"], open_info["price"],
                            _date(last, len(ich) - 1), price, open_info["size"],
                            pnl, pnl_pct, (len(ich) - 1) - open_info["i"], "end_of_backtest"))
        equity_curve.append(start_equity + realized)

    return trades, equity_curve


class Backtester:
    def __init__(self, cfg, data, logger, days: int = 700):
        self.cfg = cfg
        self.data = data
        self.log = logger
        self.days = days
        self.needed = min_required_candles(cfg.ichimoku.span_b_periods, cfg.ichimoku.displacement)

    def run_market(self, coin: str) -> tuple[list[Trade], list[float]]:
        df = self.data.fetch_daily(coin, lookback_days=self.days,
                                   drop_incomplete=self.cfg.trading.only_completed_candles)
        start_equity = self.cfg.risk.account_equity_usd
        if len(df) < self.needed:
            self.log.warning("%s: only %d candles, need %d -- skipping.", coin, len(df), self.needed)
            return [], [start_equity]
        ich = compute_ichimoku(
            df,
            conversion_periods=self.cfg.ichimoku.conversion_periods,
            base_periods=self.cfg.ichimoku.base_periods,
            span_b_periods=self.cfg.ichimoku.span_b_periods,
            displacement=self.cfg.ichimoku.displacement,
        )
        risk = RiskManager.from_config(self.cfg.risk, self.cfg.trading.max_leverage)
        return replay_history(coin, ich, risk, self.cfg.risk.min_signal_confidence, self.log)

    def run(self) -> dict:
        results = {}
        for coin in self.cfg.trading.markets:
            trades, eq = self.run_market(coin)
            results[coin] = {
                "trades": trades,
                "equity_curve": eq,
                "metrics": compute_metrics(trades, eq, self.cfg.risk.account_equity_usd),
            }
        return results


def _print_report(results: dict, start_equity: float, days: int) -> None:
    print(f"\nBacktest over ~{days} candles per market "
          f"(each simulated independently with a ${start_equity:,.0f} account)")
    print("Past performance does not predict future results.\n")
    header = f"{'MARKET':<7}{'TRADES':>7}{'WIN%':>7}{'RET%':>8}{'AVGW%':>7}{'AVGL%':>7}{'PF':>6}{'MAXDD%':>8}{'BARS':>6}"
    print(header)
    print("-" * len(header))
    all_trades = []
    for coin, r in results.items():
        m = r["metrics"]
        all_trades.extend(r["trades"])
        pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
        print(f"{coin:<7}{m['trades']:>7}{m['win_rate']*100:>7.0f}{m['total_return_pct']:>8.2f}"
              f"{m['avg_win_pct']:>7.2f}{m['avg_loss_pct']:>7.2f}{pf:>6}{m['max_drawdown_pct']:>8.2f}"
              f"{m['avg_bars_held']:>6.0f}")
    n = len(all_trades)
    wins = sum(1 for t in all_trades if t.pnl > 0)
    total_pnl = sum(t.pnl for t in all_trades)
    print("-" * len(header))
    print(f"POOLED: {n} trades, {wins} wins "
          f"({(wins/n*100) if n else 0:.0f}% win rate), total PnL ${total_pnl:.2f} "
          f"(per-market independent accounts)\n")


def main() -> int:
    import argparse

    from ichibot.config import load_config, ConfigError
    from ichibot.logging_setup import setup_logging
    from ichibot.market_data import HyperliquidData, MarketDataError

    parser = argparse.ArgumentParser(description="ichibot backtest")
    parser.add_argument("--days", type=int, default=700, help="candles of history to fetch per market")
    args = parser.parse_args()

    log = setup_logging()
    log.setLevel(logging.WARNING)   # keep per-candle chatter quiet; we want the report

    try:
        cfg = load_config("Config.yaml", ".env")
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    try:
        data = HyperliquidData()
    except MarketDataError as exc:
        print(f"Market data unavailable: {exc}")
        return 1

    bt = Backtester(cfg, data, log, days=args.days)
    results = bt.run()
    _print_report(results, cfg.risk.account_equity_usd, args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())