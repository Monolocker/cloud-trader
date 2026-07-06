"""Live order executor (Milestone 8).

Places REAL orders on Hyperliquid via hyperliquid-python-sdk (v0.24). It mirrors
DryRunExecutor's interface (process / reconcile / commit / positions) so the engine
drives either one unchanged.

Defense in depth below. Every layer is independent, so no single failure is catastrophic:
  1. Master switch    -- main.py only constructs this when ENABLE_LIVE_TRADING=true AND --live.
  2. Long-only        -- opens are ALWAYS market_open(is_buy=True); we never short.
  3. Reconciliation   -- exchange state fetched and compared to local before acting;
                         ANY divergence HALTS the run (raises LiveTradingError). We never
                         trade on a state we don't trust.
  4. Confirmation     -- main.py requires a typed 'YES' before the first live order.
  5. Hard size cap    -- max_order_usd clamps every order's notional, independent of the
                         risk manager, plus a 1x ceiling (total notional <= equity).
The RiskManager's hard stop-loss is honored in _manage_open exactly as in dry-run.
"""

from __future__ import annotations

from ichibot.risk import Position, RiskManager
from ichibot.signals import SignalResult


class LiveTradingError(Exception):
    """Raised to HALT live trading (e.g. state divergence). Never swallowed."""


class LiveExecutor:
    def __init__(self, risk: RiskManager, logger, exchange, info, account_address: str,
                 sz_decimals: dict, store, max_order_usd: float = 25.0):
        self.risk = risk
        self.log = logger
        self.exchange = exchange
        self.info = info
        self.account_address = account_address
        self.sz_decimals = sz_decimals
        self.store = store
        self.max_order_usd = max_order_usd
        self.positions = store.load() if store else {}
        self.realized_pnl = 0.0

    # --- reconciliation (guardrail 3) --------------------------------------
    def reconcile(self) -> None:
        """Fetch live positions and compare to local state. HALT on any divergence."""
        state = self.info.user_state(self.account_address)
        live = {}
        for ap in state.get("assetPositions", []):
            p = ap.get("position", {})
            coin = p.get("coin")
            szi = float(p.get("szi", 0) or 0)
            if coin and szi != 0:
                live[coin] = szi

        problems = []
        for coin in set(self.positions) | set(live):
            local_sz = self.positions[coin].size_units if coin in self.positions else 0.0
            live_sz = live.get(coin, 0.0)
            if live_sz < 0:
                problems.append(f"{coin}: exchange shows SHORT {live_sz} (not long-only)")
            tol = max(1e-6, 0.01 * max(abs(local_sz), abs(live_sz)))
            if abs(local_sz - live_sz) > tol:
                problems.append(f"{coin}: local {local_sz} vs exchange {live_sz}")

        if problems:
            raise LiveTradingError("Local/exchange state diverged, HALTING: " + "; ".join(problems))
        self.log.info("[LIVE] Reconciled OK: %d open position(s) match the exchange.", len(live))

    # --- executor interface ------------------------------------------------
    def current_exposure_usd(self) -> float:
        return sum(p.notional_usd for p in self.positions.values())

    def _round_size(self, coin: str, sz: float) -> float:
        return round(sz, self.sz_decimals.get(coin, 4))

    def process(self, coin: str, price: float, signal: SignalResult) -> str:
        if coin in self.positions:
            return self._manage_open(coin, price, signal)
        return self._consider_entry(coin, price, signal)

    def _manage_open(self, coin: str, price: float, signal: SignalResult) -> str:
        pos = self.positions[coin]
        self.risk.update_trailing_peak(pos, price)
        exit_dec = self.risk.evaluate_exit(pos, price)     # hard stop / tp / trailing -- always honored
        if exit_dec.should_exit:
            return self._close(coin, exit_dec.reason)
        if signal.exit_recommended:
            return self._close(coin, "signal:" + ",".join(signal.bearish_signals))
        self.log.info("[LIVE] HOLD %-6s @ %.4f (entry %.4f, stop %.4f)", coin, price, pos.entry_price, pos.stop_price)
        return "hold"

    def _consider_entry(self, coin: str, price: float, signal: SignalResult) -> str:
        if not signal.entry_recommended:
            return "none"
        decision = self.risk.size_position(coin, price, signal.confidence, self.current_exposure_usd())
        if not decision.approved:
            self.log.info("[LIVE] ENTRY SKIPPED %-6s: %s", coin, decision.reason)
            return "entry_rejected"

        # Guardrail 5: clamp notional by the hard cap AND the 1x headroom.
        headroom_1x = self.risk.account_equity_usd - self.current_exposure_usd()
        notional = min(decision.notional_usd, self.max_order_usd, headroom_1x)
        if notional < self.risk.min_order_usd:
            self.log.warning("[LIVE] ENTRY SKIPPED %-6s: capped notional $%.2f below minimum $%.2f",
                             coin, notional, self.risk.min_order_usd)
            return "entry_rejected"

        sz = self._round_size(coin, notional / price)
        if sz <= 0:
            self.log.warning("[LIVE] ENTRY SKIPPED %-6s: size rounds to 0", coin)
            return "entry_rejected"

        self.log.warning("[LIVE] MARKET BUY %-6s sz=%s (~$%.2f)", coin, sz, notional)
        resp = self.exchange.market_open(coin, True, sz)   # is_buy=True ALWAYS (long-only)
        fill = self._parse_fill(resp)
        if fill is None:
            self.log.error("[LIVE] ORDER FAILED %-6s: %s", coin, resp)
            return "entry_failed"

        fill_px, filled_sz = fill
        pos = Position.from_decision(decision)
        pos.entry_price = fill_px
        pos.size_units = filled_sz
        pos.notional_usd = fill_px * filled_sz
        pos.peak_price = fill_px
        pos.stop_price = fill_px * (1 - self.risk.stop_loss_frac)
        pos.take_profit_price = fill_px * (1 + self.risk.take_profit_frac) if self.risk.take_profit_frac > 0 else None
        self.positions[coin] = pos
        self._save()
        self.log.warning("[LIVE] OPENED %-6s @ %.4f sz=%s stop=%.4f", coin, fill_px, filled_sz, pos.stop_price)
        return "opened"

    def _close(self, coin: str, reason: str) -> str:
        pos = self.positions[coin]
        self.log.warning("[LIVE] MARKET CLOSE %-6s (reason %s)", coin, reason)
        resp = self.exchange.market_close(coin)
        fill = self._parse_fill(resp)
        if fill is None:
            self.log.error("[LIVE] CLOSE FAILED %-6s: %s -- position still OPEN, will retry next run", coin, resp)
            return "close_failed"
        fill_px, _ = fill
        pnl = (fill_px - pos.entry_price) * pos.size_units
        self.realized_pnl += pnl
        del self.positions[coin]
        self._save()
        self.log.warning("[LIVE] CLOSED %-6s @ %.4f pnl $%.2f reason %s", coin, fill_px, pnl, reason)
        return f"closed:{reason}"

    @staticmethod
    def _parse_fill(resp):
        """Return (avg_px, total_sz) if the order filled, else None."""
        try:
            if resp.get("status") != "ok":
                return None
            for s in resp["response"]["data"]["statuses"]:
                if "filled" in s:
                    f = s["filled"]
                    return float(f["avgPx"]), float(f["totalSz"])
                if "error" in s:
                    return None
            return None
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    def _save(self) -> None:
        if self.store:
            self.store.save(self.positions)

    def commit(self) -> None:
        self._save()