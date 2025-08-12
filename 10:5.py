import time
import logging
import pandas as pd
import numpy as np
import ccxt  # Library for connecting to exchanges




# --- CONFIGURATION ---
CONFIG = {
    # --- Exchange & API ---
    "exchange": "bybit",  # CCXT exchange ID
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",


    # --- Strategy Parameters ---
    "symbol": "ETH/USDT",
    "timeframe": "1m",
    "capital_base": 150,       # Starting capital in USDT
    "risk_per_trade_usd": 7.0,  # Max dollar amount per trade
    "take_profit_1_percent": 1.5,
    "take_profit_2_percent": 2.5,
    "profit_lock_stop_loss_percent": 0.5,
    "fast_ma_period": 5,        # SMA
    "slow_ma_period": 10,       # EMA
    "volatility_ma_period": 50, # EMA for volatility filter
    "volatility_threshold_percent": 0.1  # Min % distance between slow MA and volatility MA
}




# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(agent_name):
    """Creates a logger with a specific agent name."""
    return logging.getLogger(agent_name)




# --- AGENT 1: DATA HANDLER ---
class DataHandler:
    def __init__(self, exchange):
        self.logger = get_logger(self.__class__.__name__)
        self.exchange = exchange
        self.logger.info(f"Initialized for {self.exchange.id} exchange.")


    def get_latest_bars(self, symbol, timeframe, n_bars=100):
        """Fetches the latest N bars from the exchange."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=n_bars)
            if not ohlcv:
                self.logger.warning("No OHLCV data returned.")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except ccxt.BaseError as e:
            self.logger.error(f"CCXT error fetching data: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching data: {e}")
        return None




# --- AGENT 2: STRATEGY & SIGNAL GENERATOR ---
class Strategy:
    def __init__(self, config):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        self.logger.info("Strategy agent initialized.")


    def _calculate_indicators(self, data):
        df = data.copy()
        df['sma_fast'] = df['close'].rolling(window=self.config['fast_ma_period']).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config['slow_ma_period'], adjust=False).mean()
        df['ema_volatility'] = df['close'].ewm(span=self.config['volatility_ma_period'], adjust=False).mean()
        return df


    def generate_signal(self, data):
        if len(data) < self.config['volatility_ma_period']:
            self.logger.warning("Not enough data to generate signals.")
            return None


        df = self._calculate_indicators(data)
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]


        volatility_spread = abs(last_row['ema_slow'] - last_row['ema_volatility']) / last_row['ema_slow'] * 100
        if volatility_spread < self.config['volatility_threshold_percent']:
            self.logger.info(f"Market too flat (spread: {volatility_spread:.2f}%). No trade.")
            return None


        buy_signal = (prev_row['ema_slow'] <= prev_row['sma_fast']) and (last_row['ema_slow'] > last_row['sma_fast'])
        exit_signal = (prev_row['ema_slow'] >= prev_row['sma_fast']) and (last_row['ema_slow'] < last_row['sma_fast'])


        if buy_signal:
            self.logger.info(f"BUY SIGNAL at price {last_row['close']:.5f}")
            return {'type': 'BUY', 'price': last_row['close']}


        if exit_signal:
            self.logger.info(f"EXIT SIGNAL at price {last_row['close']:.5f}")
            return {'type': 'EXIT', 'price': last_row['close']}


        return None




# --- AGENT 3: PORTFOLIO & RISK MANAGER ---
class PortfolioManager:
    def __init__(self, config, executor):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        self.executor = executor
        self.cash = self.config['capital_base']
        self.position = None
        self.logger.info(f"Portfolio initialized with ${self.cash:,.2f}")


    def on_signal(self, signal):
        if signal['type'] == 'BUY' and self.position is None:
            self.execute_buy(signal)
        elif signal['type'] == 'EXIT' and self.position is not None:
            self.execute_sell(signal, "MA Crossover Exit")


    def execute_buy(self, signal):
        entry_price = signal['price']
        max_trade_amount = self.config['risk_per_trade_usd']
        quantity = max_trade_amount / entry_price


        if self.cash < (quantity * entry_price):
            self.logger.error("Insufficient cash for trade.")
            return


        self.position = {
            'entry_price': entry_price,
            'quantity': quantity,
            'tp1_price': entry_price * (1 + self.config['take_profit_1_percent'] / 100),
            'tp2_price': entry_price * (1 + self.config['take_profit_2_percent'] / 100),
            'profit_lock_sl_price': entry_price * (1 + self.config['profit_lock_stop_loss_percent'] / 100),
            'tp1_hit': False,
        }


        self.executor.create_market_buy_order(self.config['symbol'], quantity)
        self.cash -= quantity * entry_price
        self.logger.info(f"EXECUTED BUY: {quantity:.6f} {self.config['symbol']} at ${entry_price:.5f}")


    def execute_sell(self, signal, reason=""):
        if not self.position:
            return
        exit_price = signal['price']
        qty = self.position['quantity']


        self.executor.create_market_sell_order(self.config['symbol'], qty)
        pnl = (exit_price - self.position['entry_price']) * qty
        self.cash += qty * exit_price
        self.logger.info(f"EXECUTED SELL ({reason}): {qty:.6f} at ${exit_price:.5f} | PnL: ${pnl:.2f} | Cash: ${self.cash:,.2f}")
        self.position = None


    def check_trade_management(self, current_price):
        if not self.position:
            return


        if not self.position['tp1_hit'] and current_price >= self.position['tp1_price']:
            self.logger.info("TP1 HIT!")
            qty_to_sell = self.position['quantity'] / 2
            self.executor.create_market_sell_order(self.config['symbol'], qty_to_sell)
            pnl = (self.position['tp1_price'] - self.position['entry_price']) * qty_to_sell
            self.cash += qty_to_sell * self.position['tp1_price']
            self.position['quantity'] -= qty_to_sell
            self.position['tp1_hit'] = True
            self.logger.info(f"Took partial profit of ${pnl:.2f}. Remaining qty: {self.position['quantity']:.6f}")
            self.logger.info(f"Stop loss moved to ${self.position['profit_lock_sl_price']:.5f}")


        if self.position['tp1_hit'] and current_price <= self.position['profit_lock_sl_price']:
            self.execute_sell({'price': self.position['profit_lock_sl_price']}, "Profit Lock SL")
            return


        if self.position['tp1_hit'] and current_price >= self.position['tp2_price']:
            self.execute_sell({'price': self.position['tp2_price']}, "TP2 Hit")
            return




# --- AGENT 4: EXECUTION HANDLER ---
class ExecutionHandler:
    def __init__(self, exchange):
        self.logger = get_logger(self.__class__.__name__)
        self.exchange = exchange
        self.logger.info("Execution handler ready.")


    def create_market_buy_order(self, symbol, quantity):
        self.logger.info(f"MARKET BUY {quantity:.6f} {symbol}")
        # Uncomment for live trading:
        # return self.exchange.create_market_buy_order(symbol, quantity)


    def create_market_sell_order(self, symbol, quantity):
        self.logger.info(f"MARKET SELL {quantity:.6f} {symbol}")
        # Uncomment for live trading:
        # return self.exchange.create_market_sell_order(symbol, quantity)




# --- AGENT 5: ORCHESTRATOR ---
class Orchestrator:
    def __init__(self, config):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config


        exchange_class = getattr(ccxt, config['exchange'])
        self.exchange = exchange_class({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
            'enableRateLimit': True
        })


        self.data_handler = DataHandler(self.exchange)
        self.executor = ExecutionHandler(self.exchange)
        self.portfolio = PortfolioManager(config, self.executor)
        self.strategy = Strategy(config)
        self.logger.info("Orchestrator ready for live trading.")


    def run(self):
        self.logger.info("Starting live trading system...")
        self.logger.warning("Execution calls are commented out for safety.")


        while True:
            try:
                latest_data = self.data_handler.get_latest_bars(self.config['symbol'], self.config['timeframe'])
                if latest_data is None or latest_data.empty:
                    self.logger.warning("No data received. Retrying in 60s...")
                    time.sleep(60)
                    continue


                current_price = latest_data.iloc[-1]['close']


                if self.portfolio.position:
                    self.portfolio.check_trade_management(current_price)


                signal = self.strategy.generate_signal(latest_data)
                if signal:
                    self.portfolio.on_signal(signal)


                time.sleep(60)


            except KeyboardInterrupt:
                self.logger.info("System shutdown requested by user.")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                time.sleep(60)




if __name__ == "__main__":
    if CONFIG['api_key'] == "YOUR_API_KEY" or CONFIG['api_secret'] == "YOUR_API_SECRET":
        print("!!! Please set your API key and secret in CONFIG before running.")
    else:
        system = Orchestrator(CONFIG)
        system.run()