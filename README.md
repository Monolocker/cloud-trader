# cloud-trader
An Ichimoku Cloud trading bot for [Hyperliquid](https://app.hyperliquid.xyz/trade) perps, written in Python. It evaluates a rule-based Ichimoku strategy on daily candles, sizes positions through a risk manager, and can run either as a paper-trading simulator or against the live exchange behind layered safety guardrails. 

This is a personal, educational project built to learn/strengthen software engineering and systematic trading concepts such as: 
- API Integration
- System Design / Behavior 
- Backtesting
- Signal Detection 
- Live Order Execution

## What it is / what it is not 
**It is:** 
    - A rule-based (non-ML) strategy engine with a full backtester 
    - A dry-run-by-default bot that paper-trades and persists state locally
    - A live executor for Hyperliquid perps, gated behind multiple independent safety layers
    - A test-covered codebase (~120 tests) built over the course of major/minor milestones

**It is NOT:** 
    - A money-making guarantee, a signal service, or financial advice
    - A high-frequency or leveraged system. Trades 1d, 1x
    - Production-hardened infrastructure. It is a learning project, run and reviewed by hand

## Strategy 
The strategy is a long-only trend-follower built on the Ichimoku cloud. The following are subject to change:
    - **Indicator:** Ichimoku Cloud with "doubled" settings. Conversion `20`, base `60`, Span B `120`, displacement `30`. Applied to daily candles
    - **Direction:** Long-only 
    - **Leverage:** 1x
    - **Markets:** BTC, ETH, SOL, HYPE perpetuals
    - **Signals:** 5 core signals, including additional pattern signals, and a continuation re-entry signal
    - **Exits:** Structural bearish signals, including a hard stop-loss enforced independently by the risk manager

## Validated Profile 
The strategy has been backtested across multiple markets and time samples with a buy-and-hold benchmark for context. Its character is defensive: it is a risk-managed trend-follower that sits in cash when signals are absent, caps exposure per position, and prioritizes capital preservation in down and choppy markets. It has shown small max drawdowns (roughly 2–5%) relative to holding the underlying assets through the same periods in such market environments.

Backtests are saved as timestamped `.txt` / `.json` pairs under `results/` for later comparison. 

## Architecture
The code is organized as a flat `ichibot/` package, with a thin `main.py` launcher at the repository root.

| Module | Responsibility |
| --- | --- | 
| `config.py` | Load + validate `config.yaml` and secrets from `.env` | 
| `logging_setup.py` | Rotating-file + console logging |
| `market_data.py` | Read-only Hyperliquid candle fetching → DataFrame | 
| `ichimoku.py` | Compute Ichimoku lines, cloud, and leading spans | 
| `signals.py` | Detect entry/exit signals. Combine into a weighted decision | 
| `risk.py` | Capital-based position sizing, stop-loss, exposure caps | 
| `executor_dryrun.py` | opens/tracks/closes simulated positions on paper, persisted to `data/positions.json` | 
| `executor_live.py` | Live executor; places real orders via the Hyperliquid SDK, with reconciliation and guardrails | 
| `engine.py` | Orchestration/scheduler: fetch → indicators → signals → executor, once or in a daily loop. |
| `backtest.py` | Replay the strategy over history; metrics, benchmark, per-signal attribution, saved results |

## Requirements
- Python 3.13 
- pip package manager 
- Git (for cloning repository)

## Python Dependencies
```
# Core dependencies
python-dotenv>=1.0.0
PyYAML>=6.0
pandas>=2.0
hyperliquid-python-sdk>=0.23.0

# Development / testing
pytest>=8.0
```

## Exchange Requirements 
- Hyperliquid account with API access
- Perpetual trading enabled 
- Adequate amount of USDC 

## Setup
```
git clone https://github.com/monolocker/cloud-trader.git
cd cloud-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Create a `.env` file in the project root (this file is gitignored and must never be committed):
```
ENABLE_LIVE_TRADING=false
HYPERLIQUID_PRIVATE_KEY=<your-agent-wallet-private-key>
HYPERLIQUID_ACCOUNT_ADDRESS=<your-main-wallet-address>
```

## Usage 
**Dry run (default and never places real orders):**
```
python main.py
```
**Run continuously, once shortly after each 00:00 UTC:**
```
python main.py --loop
```
**Backtest over the last N daily candles per market:**
```
python -m ichibot.backtest --days 1500
```
**Live trading (testnet first, use faucet):**
```
python main.py --live --testnet --max-order-usd 12
```
**Live trading (mainnet):**
```
python main.py --live --max-order-usd <notional_cap>
```
Live trading requires **both** `ENABLE_LIVE_TRADING=true` in the environment **and** the `--live` flag. `--testnet` routes execution to the Hyperliquid testnet and loads `.env.testnet`. `--max-order-usd` caps the notional of any single order.

## Safety / Guardrails
Live trading is protected by 5 independent layers, so no single failure leads to a catastrophic result: 
1. **Master switch:** Live trading requires both the `ENABLE_LIVE_TRADING` environment flag and the `--live` command-line flag
2. **Long-only enforcement:** Order placement only ever opens long positions
3. **Reconciliation:** Before acting, the bot fetches actual positions from the exchange and compares them to local state. Any divergence halts the run rather than trading on an untrusted view
4. **Typed confirmation:** The first arming of live trading requires a typed `YES` after a banner showing the network, account, and size cap
5. **Hard size cap:** Every order's notional is anchored by `--max-order-usd` and by a 1x total-exposure ceiling, independent of the risk manager. Hard stop-loss is always honored and never suppressed. 

## Development + Testing 
This project was built incrementally across defined major and minor milestones, each verified before moving on. The test suite (120 tests, `pytest`) covers config, market data, indicators, signals, risk sizing, both executors, the engine, and the backtester. This includes guardrail and reconciliation logic for the live path. 
```
python -m pytest -q
```
Run artifacts are kept out of version control: `data/` (positions/state), `logs/`, and `results/` (saved backtests) are gitignored, as are all `.env*` files.

## Disclaimer 
This software is experimental and provided for **educational purposes only**. This is **not financial advice**. Trading cryptocurrency perpetuals carries substantial risk, including the total loss of funds. Use entirely at your own risk. 



