### 10-5 Trading Bot (10 EMA / 5 SMA)

A minimal live-trading scaffold using CCXT with a simple moving average crossover and volatility filter. The code is structured into agents for data, strategy, portfolio, execution, and orchestration.

Important: Order execution is logged only. Live order methods are commented out for safety in `ExecutionHandler`. Uncomment them to place real orders once you understand the risks.

---

### Features
- **Exchange**: Configurable via CCXT (default: `bybit`).
- **Strategy**: 5-period SMA vs 10-period EMA crossover with a volatility filter.
- **Risk**: Fixed dollar risk per trade, partial take-profit, profit lock stop-loss.
- **Config**: All parameters are overrideable via environment variables.

---

### Requirements
- Python 3.11+
- Dependencies in `requirements.txt`: `ccxt`, `pandas`, `numpy`

---

### Environment Variables
The script reads configuration from environment variables. A sample `.env.example` is provided; copy it to `.env` and edit values.

- **EXCHANGE**: CCXT exchange id (e.g., `bybit`)
- **API_KEY**: Your exchange API key
- **API_SECRET**: Your exchange API secret
- **SYMBOL**: Trading pair (e.g., `ETH/USDT`)
- **TIMEFRAME**: CCXT timeframe (e.g., `1m`)
- **CAPITAL_BASE**: Starting cash in USDT
- **RISK_PER_TRADE_USD**: Max dollar amount per trade
- **TAKE_PROFIT_1_PERCENT**: First TP percent
- **TAKE_PROFIT_2_PERCENT**: Second TP percent
- **PROFIT_LOCK_STOP_LOSS_PERCENT**: Stop-loss after TP1
- **FAST_MA_PERIOD**: Fast SMA period
- **SLOW_MA_PERIOD**: Slow EMA period
- **VOLATILITY_MA_PERIOD**: EMA period for volatility filter
- **VOLATILITY_THRESHOLD_PERCENT**: Minimum % spread between slow EMA and volatility EMA
- **DEFAULT_TYPE**: Exchange market type (e.g., `spot`, `linear`, `inverse`), used for some exchanges like Bybit
- **USE_SANDBOX**: `true` to enable exchange sandbox/testnet if supported
- **SANDBOX_API_BASE**: Override base REST URL for all API endpoints (for exchanges without `set_sandbox_mode`)
- **SANDBOX_PUBLIC_URL**: Override REST public endpoint only
- **SANDBOX_PRIVATE_URL**: Override REST private endpoint only
- **SANDBOX_WS_URL**: Override WebSocket endpoint

See `.env.example` for defaults and placeholders.

---

### Local Setup
1. Create and activate a virtual environment (optional but recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
3. Configure environment:
- Copy `env.example` to `.env` (or export directly), then export them:
```bash
cp env.example .env
export $(grep -v '^#' .env | xargs)
```
4. Run the bot:
```bash
python 10:5.py
```

If you see the message asking to set API credentials, ensure `API_KEY` and `API_SECRET` are set.

---

### Docker (local or cloud)
Build the image:
```bash
docker build -t tradingbot:latest .
```
Run with your env file:
```bash
docker run --rm --name tradingbot --env-file ./.env tradingbot:latest
```

### Sandbox / Testnet (cloud-safe)
- Default behavior uses sandbox if available: `USE_SANDBOX=true` in `env.example`.
- Some exchanges require specific testnet URLs. You can override using:
  - `SANDBOX_API_BASE` to set all endpoints at once, or
  - `SANDBOX_PUBLIC_URL` / `SANDBOX_PRIVATE_URL` / `SANDBOX_WS_URL` separately.
- Example (Bybit testnet spot as illustration; verify current URLs in exchange docs):
```bash
SANDBOX_PUBLIC_URL=https://api-testnet.bybit.com
SANDBOX_PRIVATE_URL=https://api-testnet.bybit.com
```

Notes:
- The container runs `python 10:5.py` by default.
- Adjust `.env` to change parameters without rebuilding.
- Set `USE_SANDBOX=false` only when you are ready for live trading.

### CI/CD (GitHub Actions)
- On push to `main`, a Docker image is built and pushed to GitHub Container Registry under `ghcr.io/<owner>/10-5-tradingbot`.
- To run the image on any cloud (e.g., ECS, Cloud Run, ACI):
  1. Create a secret store containing your `.env` values.
  2. Mount or inject environment variables.
  3. Use the image tag from GHCR.

---

### Live Trading Safety
- In `ExecutionHandler`, live order methods are commented out:
```python
# return self.exchange.create_market_buy_order(symbol, quantity)
# return self.exchange.create_market_sell_order(symbol, quantity)
```
Uncomment only when you fully understand the risks and your exchange permissions. Use testnet/sandbox if available.

---

### Troubleshooting
- **No OHLCV data**: Ensure the symbol and timeframe are supported by your exchange.
- **Rate limit errors**: CCXT `enableRateLimit` is enabled, but you may still need to slow the loop.
- **Credentials error**: Verify `API_KEY`, `API_SECRET`, and exchange settings.

---

### License
MIT
