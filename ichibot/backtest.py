"""Backtest the Ichimoku strategy over a long history (Backtest milestone).

Replays the exact current live pipeline: compute_ichimoku -> evaluate_signals ->
RiskManager -> DryRunExecutor -- candle-by-candle over a long lookback, and
reports per-market metrics (trades, win rate, return, profit factor, drawdown).

Each market is isolated upon simulation, as if it had the full account. This
answers "does the strategy have edge on this instrument?". A single shared-
portfolio simulation (cross-market exposure on one account) can be a later
enhancement.

This is a measurement tool. It places no orders.

Run:  python -m ichibot.backtest --days 700
"""

from __future__ import annotations
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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
    entry_signals: tuple = ()


def max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]; mdd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def compute_metrics(trades, equity_curve, start_equity):
    n = len(trades); wins = [t for t in trades if t.pnl > 0]; losses = [t for t in trades if t.pnl <= 0]
    gp = sum(t.pnl for t in wins); gl = -sum(t.pnl for t in losses)
    fe = equity_curve[-1] if equity_curve else start_equity
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    return {"trades": n, "win_rate": (len(wins) / n) if n else 0.0,
            "total_return_pct": (fe / start_equity - 1) * 100,
            "avg_win_pct": (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0.0,
            "avg_loss_pct": (sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0.0,
            "profit_factor": pf, "max_drawdown_pct": max_drawdown(equity_curve) * 100,
            "avg_bars_held": (sum(t.bars_held for t in trades) / n) if n else 0.0, "final_equity": fe}


def signal_attribution(trades):
    agg = {}
    for t in trades:
        for sig in t.entry_signals:
            a = agg.setdefault(sig, {"trades": 0, "wins": 0, "pnl": 0.0})
            a["trades"] += 1
            a["pnl"] += t.pnl
            if t.pnl > 0:
                a["wins"] += 1
    return agg


def replay_history(coin, ich, risk, min_confidence, logger):
    ex = DryRunExecutor(risk, logger, store=None); start = risk.account_equity_usd
    trades = []; eq = [start]; realized = 0.0; oi = {}

    def d(row, i):
        return str(row["time"].date()) if "time" in ich.columns else str(i)

    for i in range(1, len(ich)):
        w = ich.iloc[: i + 1]; row = w.iloc[-1]; price = float(row["close"])
        sig = evaluate_signals(w, min_confidence)
        action = ex.process(coin, price, sig)
        if action == "opened":
            pos = ex.positions[coin]
            oi = {"date": d(row, i), "price": pos.entry_price, "size": pos.size_units,
                  "i": i, "signals": tuple(sig.bullish_signals)}
        elif action.startswith("closed:"):
            reason = action.split(":", 1)[1]
            pnl = (price - oi["price"]) * oi["size"]; pct = (price / oi["price"] - 1) * 100
            realized += pnl
            trades.append(Trade(coin, oi["date"], oi["price"], d(row, i), price, oi["size"],
                                pnl, pct, i - oi["i"], reason, entry_signals=oi["signals"]))
            eq.append(start + realized)

    if coin in ex.positions:
        last = ich.iloc[-1]; price = float(last["close"]); ex.close_position(coin, price, "end_of_backtest")
        pnl = (price - oi["price"]) * oi["size"]; pct = (price / oi["price"] - 1) * 100; realized += pnl
        trades.append(Trade(coin, oi["date"], oi["price"], d(last, len(ich) - 1), price, oi["size"],
                            pnl, pct, (len(ich) - 1) - oi["i"], "end_of_backtest", entry_signals=oi["signals"]))
        eq.append(start + realized)
    return trades, eq


class Backtester:
    def __init__(self, cfg, data, logger, days=700):
        self.cfg = cfg; self.data = data; self.log = logger; self.days = days
        self.needed = min_required_candles(cfg.ichimoku.span_b_periods, cfg.ichimoku.displacement)

    def run_market(self, coin):
        df = self.data.fetch_daily(coin, lookback_days=self.days,
                                   drop_incomplete=self.cfg.trading.only_completed_candles)
        se = self.cfg.risk.account_equity_usd
        if len(df) < self.needed:
            return [], [se], 0.0
        ich = compute_ichimoku(df, conversion_periods=self.cfg.ichimoku.conversion_periods,
                               base_periods=self.cfg.ichimoku.base_periods,
                               span_b_periods=self.cfg.ichimoku.span_b_periods,
                               displacement=self.cfg.ichimoku.displacement)
        risk = RiskManager.from_config(self.cfg.risk, self.cfg.trading.max_leverage)
        trades, eq = replay_history(coin, ich, risk, self.cfg.risk.min_signal_confidence, self.log)
        return trades, eq, buy_and_hold_return_pct(ich)

    def run(self):
        r = {}
        for c in self.cfg.trading.markets:
            t, e, bh = self.run_market(c)
            r[c] = {"trades": t, "equity_curve": e, "buy_hold_pct": bh,
                    "metrics": compute_metrics(t, e, self.cfg.risk.account_equity_usd)}
        return r


def _format_report(results, start_equity, days) -> str:
    lines = [f"Backtest over ~{days} candles per market "
             f"(each simulated independently with a ${start_equity:,.0f} account)",
             "Past performance does not predict future results.", ""]
    header = (f"{'MARKET':<7}{'TRADES':>7}{'WIN%':>7}{'RET%':>8}{'BH%':>8}{'EDGE%':>8}"
              f"{'AVGW%':>7}{'AVGL%':>7}{'PF':>6}{'MAXDD%':>8}{'BARS':>6}")
    lines += [header, "-" * len(header)]
    all_trades = []
    for coin, r in results.items():
        m = r["metrics"]; all_trades.extend(r["trades"])
        bh = r["buy_hold_pct"]
        edge = m["total_return_pct"] - bh
        pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
        lines.append(f"{coin:<7}{m['trades']:>7}{m['win_rate']*100:>7.0f}{m['total_return_pct']:>8.2f}"
                     f"{bh:>8.2f}{edge:>8.2f}"
                     f"{m['avg_win_pct']:>7.2f}{m['avg_loss_pct']:>7.2f}{pf:>6}{m['max_drawdown_pct']:>8.2f}"
                     f"{m['avg_bars_held']:>6.0f}")
    n = len(all_trades); wins = sum(1 for t in all_trades if t.pnl > 0); total = sum(t.pnl for t in all_trades)
    lines += ["-" * len(header),
              f"POOLED: {n} trades, {wins} wins ({(wins/n*100) if n else 0:.0f}% win rate), "
              f"total PnL ${total:.2f} (per-market independent accounts)"]
    return "\n".join(lines)


def _format_attribution(results) -> str:
    all_trades = []
    for r in results.values():
        all_trades.extend(r["trades"])
    agg = signal_attribution(all_trades)
    lines = ["", "Entry-signal attribution (a trade counts toward every bullish signal that fired",
             "at entry; pnl is shared, so columns can sum to more than the pooled total):", ""]
    header = f"{'SIGNAL':<28}{'TRADES':>7}{'WINS':>6}{'WIN%':>7}{'PNL$':>9}"
    lines += [header, "-" * len(header)]
    for sig, a in sorted(agg.items(), key=lambda kv: kv[1]["pnl"]):
        wr = (a["wins"] / a["trades"] * 100) if a["trades"] else 0
        lines.append(f"{sig:<28}{a['trades']:>7}{a['wins']:>6}{wr:>7.0f}{a['pnl']:>9.2f}")
    return "\n".join(lines)


def _report_text(results, start_equity, days) -> str:
    return _format_report(results, start_equity, days) + "\n" + _format_attribution(results)


def _run_filename(days: int, ext: str, now: datetime | None = None) -> str:
    """Filesystem-safe result filename: backtest_<YYYY-MM-DD>_<HHMM>_<days>d.<ext>.
    No colons (some filesystems reject them). `now` is injectable for testing."""
    now = now or datetime.now()
    return f"backtest_{now:%Y-%m-%d_%H%M}_{days}d.{ext}"


def _json_safe_metrics(m: dict) -> dict:
    """Replace non-finite floats (e.g. profit_factor == inf) with None for valid JSON."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in m.items()}


def save_results(results, start_equity, days, out_dir="results"):
    """Write a timestamped .txt (human) and .json (machine) pair. Returns (txt_path, json_path)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    txt_path = out / _run_filename(days, "txt", now)
    json_path = out / _run_filename(days, "json", now)

    txt_path.write_text(_report_text(results, start_equity, days), encoding="utf-8")

    all_trades = []
    payload = {"generated_at": now.isoformat(), "days": days,
               "start_equity": start_equity, "markets": {}}
    for coin, r in results.items():
        payload["markets"][coin] = {
            "metrics": _json_safe_metrics(r["metrics"]),
            "trades": [asdict(t) for t in r["trades"]],
        }
        all_trades.extend(r["trades"])
    payload["attribution"] = signal_attribution(all_trades)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return txt_path, json_path


def main() -> int:
    import argparse
    from ichibot.config import load_config, ConfigError
    from ichibot.logging_setup import setup_logging
    from ichibot.market_data import HyperliquidData, MarketDataError

    parser = argparse.ArgumentParser(description="ichibot backtest")
    parser.add_argument("--days", type=int, default=700)
    args = parser.parse_args()

    log = setup_logging(); log.setLevel(logging.WARNING)
    try:
        cfg = load_config("Config.yaml", ".env")
    except ConfigError as exc:
        print(f"Configuration error: {exc}"); return 1
    try:
        data = HyperliquidData()
    except MarketDataError as exc:
        print(f"Market data unavailable: {exc}"); return 1

    results = Backtester(cfg, data, log, days=args.days).run()
    print(_report_text(results, cfg.risk.account_equity_usd, args.days))
    txt_path, json_path = save_results(results, cfg.risk.account_equity_usd, args.days)
    print(f"\nSaved: {txt_path}\n       {json_path}")
    return 0

def buy_and_hold_return_pct(ich) -> float:
    """Return % of buying at the first tradeable candle and holding to the last.
    Measured over the SAME window the strategy trades (after the warmup NaNs),
    so it's an apples-to-apples benchmark."""
    valid = ich[ich["cloud_top"].notna()]
    if len(valid) < 2:
        return 0.0
    first = float(valid.iloc[0]["close"])
    last = float(valid.iloc[-1]["close"])
    if first <= 0:
        return 0.0
    return (last / first - 1.0) * 100.0

if __name__ == "__main__":
    raise SystemExit(main())