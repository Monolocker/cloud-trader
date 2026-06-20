"""Risk management for the Ichimoku bot (Milestone 5).

Turns an entry recommendation into a sized, risk-checked decision, and evaluates
open positions for SL / TP / trailing-stop exits.

Sizing model: CAPITAL-BASED (fixed-fractional notional).
  position notional = max_capital_per_trade_frac * account_equity
  At 1x leverage, notional == margin (no borrowing), leverage stays <= 1x.
  Dollar risk if stopped = notional * stop_loss_frac.
(Alternative "risk-based" model: size so the stop loses a fixed % of equity
    - this module implements the capital-based model that the
    config describes.) To be implemented later
"""

from __future__ import annotations

from dataclasses import dataclass


class RiskError(Exception):
    """Raised when the risk manager is constructed with invalid parameters."""


@dataclass
class RiskDecision:
    approved: bool
    coin: str
    reason: str
    price: float
    size_units: float
    notional_usd: float
    stop_price: float | None
    take_profit_price: float | None
    dollar_risk: float

    def summary(self) -> str:
        if not self.approved:
            return f"REJECTED: {self.reason}"
        tp = f"{self.take_profit_price:.4f}" if self.take_profit_price is not None else "off"
        return (f"APPROVED notional=${self.notional_usd:.2f} "
                f"({self.size_units:.6f} units) stop={self.stop_price:.4f} "
                f"tp={tp} risk=${self.dollar_risk:.2f}")


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str                 # "" if no exit; else stop_loss / take_profit / trailing_stop
    trigger_price: float | None


@dataclass
class Position:
    coin: str
    entry_price: float
    size_units: float
    notional_usd: float
    stop_price: float
    take_profit_price: float | None
    peak_price: float           # highest price seen since entry (for trailing stop)

    @classmethod
    def from_decision(cls, d: "RiskDecision") -> "Position":
        if not d.approved:
            raise RiskError("Cannot open a position from a rejected RiskDecision")
        return cls(
            coin=d.coin, entry_price=d.price, size_units=d.size_units,
            notional_usd=d.notional_usd, stop_price=d.stop_price,
            take_profit_price=d.take_profit_price, peak_price=d.price,
        )


class RiskManager:
    """Stateless risk checks. Holds no positions; the engine/executor does that."""

    def __init__(
        self,
        account_equity_usd: float = 1000.0,
        max_capital_per_trade_frac: float = 0.10,
        max_portfolio_exposure_frac: float = 0.50,
        stop_loss_frac: float = 0.05,
        take_profit_frac: float = 0.15,
        use_trailing_stop: bool = False,
        trailing_stop_frac: float = 0.07,
        min_signal_confidence: float = 0.6,
        max_leverage: float = 1.0,
        min_order_usd: float = 10.0,
    ):
        if account_equity_usd <= 0:
            raise RiskError("account_equity_usd must be > 0")
        if not (0 < max_leverage <= 1.0):
            raise RiskError("max_leverage must be > 0 and <= 1 (this bot is 1x-only)")
        if not (0 < max_capital_per_trade_frac <= 1.0):
            raise RiskError("max_capital_per_trade_frac must be in (0, 1]")
        if not (0 < max_portfolio_exposure_frac <= 1.0):
            raise RiskError("max_portfolio_exposure_frac must be in (0, 1]")
        if max_capital_per_trade_frac > max_portfolio_exposure_frac:
            raise RiskError("per-trade cap cannot exceed portfolio exposure cap")
        if not (0 < stop_loss_frac < 1.0):
            raise RiskError("stop_loss_frac must be in (0, 1)")
        if take_profit_frac < 0:
            raise RiskError("take_profit_frac must be >= 0 (0 disables)")
        if min_order_usd < 0:
            raise RiskError("min_order_usd must be >= 0")

        self.account_equity_usd = account_equity_usd
        self.max_capital_per_trade_frac = max_capital_per_trade_frac
        self.max_portfolio_exposure_frac = max_portfolio_exposure_frac
        self.stop_loss_frac = stop_loss_frac
        self.take_profit_frac = take_profit_frac
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_frac = trailing_stop_frac
        self.min_signal_confidence = min_signal_confidence
        self.max_leverage = max_leverage
        self.min_order_usd = min_order_usd

    @classmethod
    def from_config(cls, risk_cfg, max_leverage: float) -> "RiskManager":
        return cls(
            account_equity_usd=risk_cfg.account_equity_usd,
            max_capital_per_trade_frac=risk_cfg.max_capital_per_trade_frac,
            max_portfolio_exposure_frac=risk_cfg.max_portfolio_exposure_frac,
            stop_loss_frac=risk_cfg.stop_loss_frac,
            take_profit_frac=risk_cfg.take_profit_frac,
            use_trailing_stop=risk_cfg.use_trailing_stop,
            trailing_stop_frac=risk_cfg.trailing_stop_frac,
            min_signal_confidence=risk_cfg.min_signal_confidence,
            max_leverage=max_leverage,
        )

    @property
    def per_trade_cap_usd(self) -> float:
        return self.max_capital_per_trade_frac * self.account_equity_usd

    @property
    def max_exposure_usd(self) -> float:
        return self.max_portfolio_exposure_frac * self.account_equity_usd

    def _reject(self, coin: str, price: float, reason: str) -> RiskDecision:
        return RiskDecision(False, coin, reason, price, 0.0, 0.0, None, None, 0.0)

    def size_position(
        self, coin: str, price: float, confidence: float,
        current_exposure_usd: float = 0.0,
    ) -> RiskDecision:
        """Size and risk-check a prospective long entry at `price`."""
        if price <= 0:
            return self._reject(coin, price, f"invalid price {price}")
        if confidence < self.min_signal_confidence:
            return self._reject(
                coin, price,
                f"confidence {confidence:.2f} < threshold {self.min_signal_confidence:.2f}")
        if current_exposure_usd < 0:
            return self._reject(coin, price, "current exposure cannot be negative")

        remaining = self.max_exposure_usd - current_exposure_usd
        if remaining <= 0:
            return self._reject(
                coin, price,
                f"portfolio exposure limit reached"
                f"(${current_exposure_usd:.2f} / ${self.max_exposure_usd:.2f})")

        # Take the smaller of the per-trade cap and the remaining exposure budget.
        notional = min(self.per_trade_cap_usd, remaining)
        if notional < self.min_order_usd:
            return self._reject(
                coin, price,
                f"available notional ${notional:.2f} below minimum order ${self.min_order_usd:.2f}")

        size_units = notional / price
        stop_price = price * (1 - self.stop_loss_frac)
        take_profit_price = (
            price * (1 + self.take_profit_frac) if self.take_profit_frac > 0 else None)
        dollar_risk = notional * self.stop_loss_frac

        return RiskDecision(
            approved=True, coin=coin, reason="ok", price=price,
            size_units=size_units, notional_usd=notional, stop_price=stop_price,
            take_profit_price=take_profit_price, dollar_risk=dollar_risk)

    def evaluate_exit(self, position: Position, current_price: float) -> ExitDecision:
        """Check an open long for a stop-loss / take-profit / trailing-stop exit."""
        if current_price <= position.stop_price:
            return ExitDecision(True, "stop_loss", position.stop_price)
        if position.take_profit_price is not None and current_price >= position.take_profit_price:
            return ExitDecision(True, "take_profit", position.take_profit_price)
        if self.use_trailing_stop:
            trail_price = position.peak_price * (1 - self.trailing_stop_frac)
            if current_price <= trail_price:
                return ExitDecision(True, "trailing_stop", trail_price)
        return ExitDecision(False, "", None)

    def update_trailing_peak(self, position: Position, current_price: float) -> None:
        """Raise the position's recorded peak (call once per new candle)."""
        if current_price > position.peak_price:
            position.peak_price = current_price