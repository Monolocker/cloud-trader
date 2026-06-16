"""Configuration loading and validation for the Ichimoku trading bot.

Reads non-secret settings from a YAML file (Config.yaml) and secrets / mode
flags from environment variables (loaded from a .env file).

NOTHING secret is ever read from the YAML file. Private keys live only in .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or fails validation."""


# --- Typed config sections -------------------------------------------------


@dataclass
class IchimokuConfig:
    conversion_periods: int   # Tenkan-sen
    base_periods: int         # Kijun-sen
    span_b_periods: int       # Senkou Span B
    displacement: int         # forward shift of the cloud


@dataclass
class TradingConfig:
    markets: list[str]        # e.g. ["BTC", "ETH", "SOL"]
    timeframe: str            # only "1d" supported for now
    max_leverage: float       # hard-capped at 1.0 by validation
    only_completed_candles: bool


@dataclass
class RiskConfig:
    account_equity_usd: float          # equity used for position sizing
    max_capital_per_trade_frac: float  # 0.10 == 10% of equity per trade
    max_portfolio_exposure_frac: float # 0.50 == 50% of equity deployed at once
    stop_loss_frac: float              # 0.05 == 5% below entry
    take_profit_frac: float            # 0.15 == 15% above entry (0 disables)
    use_trailing_stop: bool
    trailing_stop_frac: float          # 0.07 == trail 7% below peak
    min_signal_confidence: float       # 0..1 minimum to act on a signal


@dataclass
class AppConfig:
    trading: TradingConfig
    ichimoku: IchimokuConfig
    risk: RiskConfig
    enable_live_trading: bool
    private_key: str | None
    account_address: str | None

    def summary(self) -> dict:
        """A redacted, log-safe view of the config. Never includes secrets."""
        return {
            "mode": "LIVE" if self.enable_live_trading else "DRY_RUN",
            "markets": self.trading.markets,
            "timeframe": self.trading.timeframe,
            "max_leverage": self.trading.max_leverage,
            "ichimoku": (
                self.ichimoku.conversion_periods,
                self.ichimoku.base_periods,
                self.ichimoku.span_b_periods,
                self.ichimoku.displacement,
            ),
            "max_capital_per_trade_frac": self.risk.max_capital_per_trade_frac,
            "max_portfolio_exposure_frac": self.risk.max_portfolio_exposure_frac,
            "min_signal_confidence": self.risk.min_signal_confidence,
            "has_private_key": bool(self.private_key),
            "account_address": self.account_address,
        }


# --- Helpers ---------------------------------------------------------------


def _require(section: dict, key: str, where: str):
    """Return section[key] or raise a clear ConfigError."""
    if key not in section:
        raise ConfigError(f"Missing required key '{key}' in '{where}' section of Config.yaml")
    return section[key]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"true", "1", "yes", "on"}


def _check_frac(name: str, value: float, *, allow_zero: bool = True) -> None:
    lo = 0.0 if allow_zero else 1e-9
    if not isinstance(value, (int, float)) or value < lo or value > 1.0:
        raise ConfigError(
            f"'{name}' must be a number between {lo} and 1.0 (got {value!r})"
        )


# --- Loader ----------------------------------------------------------------


def load_config(yaml_path: str = "Config.yaml", env_path: str | None = ".env") -> AppConfig:
    """Load and validate configuration. Raises ConfigError on any problem."""
    # 1) Load secrets / mode flags from .env if present (real env vars win).
    if env_path and Path(env_path).exists():
        load_dotenv(env_path)

    # 2) Load the YAML settings file.
    p = Path(yaml_path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {yaml_path}")
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:  # defensive: malformed YAML
        raise ConfigError(f"Could not parse {yaml_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{yaml_path} must contain a top-level mapping (key: value).")

    t = _require(raw, "trading", "root")
    i = _require(raw, "ichimoku", "root")
    r = _require(raw, "risk", "root")

    cfg = AppConfig(
        trading=TradingConfig(
            markets=_require(t, "markets", "trading"),
            timeframe=_require(t, "timeframe", "trading"),
            max_leverage=float(_require(t, "max_leverage", "trading")),
            only_completed_candles=bool(_require(t, "only_completed_candles", "trading")),
        ),
        ichimoku=IchimokuConfig(
            conversion_periods=int(_require(i, "conversion_periods", "ichimoku")),
            base_periods=int(_require(i, "base_periods", "ichimoku")),
            span_b_periods=int(_require(i, "span_b_periods", "ichimoku")),
            displacement=int(_require(i, "displacement", "ichimoku")),
        ),
        risk=RiskConfig(
            account_equity_usd=float(_require(r, "account_equity_usd", "risk")),
            max_capital_per_trade_frac=float(_require(r, "max_capital_per_trade_frac", "risk")),
            max_portfolio_exposure_frac=float(_require(r, "max_portfolio_exposure_frac", "risk")),
            stop_loss_frac=float(_require(r, "stop_loss_frac", "risk")),
            take_profit_frac=float(_require(r, "take_profit_frac", "risk")),
            use_trailing_stop=bool(_require(r, "use_trailing_stop", "risk")),
            trailing_stop_frac=float(_require(r, "trailing_stop_frac", "risk")),
            min_signal_confidence=float(_require(r, "min_signal_confidence", "risk")),
        ),
        enable_live_trading=_env_bool("ENABLE_LIVE_TRADING", default=False),
        private_key=os.getenv("HYPERLIQUID_PRIVATE_KEY"),
        account_address=os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS"),
    )

    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig) -> None:
    """Enforce safety and sanity rules. Raises ConfigError on the first problem."""
    tr, ic, rk = cfg.trading, cfg.ichimoku, cfg.risk

    # Trading
    if not isinstance(tr.markets, list) or not tr.markets:
        raise ConfigError("'trading.markets' must be a non-empty list, e.g. ['BTC', 'ETH']")
    if any(not isinstance(m, str) or not m.strip() for m in tr.markets):
        raise ConfigError("Every entry in 'trading.markets' must be a non-empty string.")
    if tr.timeframe != "1d":
        raise ConfigError("Only the '1d' timeframe is supported right now.")
    if not (0 < tr.max_leverage <= 1.0):
        raise ConfigError(f"'trading.max_leverage' must be > 0 and <= 1 (got {tr.max_leverage}).")

    # Ichimoku
    for name, val in (
        ("conversion_periods", ic.conversion_periods),
        ("base_periods", ic.base_periods),
        ("span_b_periods", ic.span_b_periods),
        ("displacement", ic.displacement),
    ):
        if val <= 0:
            raise ConfigError(f"'ichimoku.{name}' must be a positive integer (got {val}).")
    if not (ic.conversion_periods < ic.base_periods < ic.span_b_periods):
        raise ConfigError(
            "Ichimoku periods should satisfy conversion < base < span_b "
            f"(got {ic.conversion_periods}, {ic.base_periods}, {ic.span_b_periods})."
        )

    # Risk
    if rk.account_equity_usd <= 0:
        raise ConfigError("'risk.account_equity_usd' must be greater than 0.")
    _check_frac("risk.max_capital_per_trade_frac", rk.max_capital_per_trade_frac, allow_zero=False)
    _check_frac("risk.max_portfolio_exposure_frac", rk.max_portfolio_exposure_frac, allow_zero=False)
    _check_frac("risk.stop_loss_frac", rk.stop_loss_frac, allow_zero=False)
    _check_frac("risk.take_profit_frac", rk.take_profit_frac, allow_zero=True)
    _check_frac("risk.trailing_stop_frac", rk.trailing_stop_frac, allow_zero=True)
    _check_frac("risk.min_signal_confidence", rk.min_signal_confidence, allow_zero=True)
    if rk.max_capital_per_trade_frac > rk.max_portfolio_exposure_frac:
        raise ConfigError(
            "'max_capital_per_trade_frac' cannot exceed 'max_portfolio_exposure_frac'."
        )
    if rk.use_trailing_stop and rk.trailing_stop_frac <= 0:
        raise ConfigError("'trailing_stop_frac' must be > 0 when 'use_trailing_stop' is true.")

    # Live-trading guardrails (defense in depth; the executor re-checks later).
    if cfg.enable_live_trading:
        if not cfg.private_key:
            raise ConfigError(
                "ENABLE_LIVE_TRADING=true but HYPERLIQUID_PRIVATE_KEY is not set in .env"
            )
        if not cfg.account_address:
            raise ConfigError(
                "ENABLE_LIVE_TRADING=true but HYPERLIQUID_ACCOUNT_ADDRESS is not set in .env"
            )
