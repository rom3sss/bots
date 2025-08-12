import time
import logging
import pandas as pd
import numpy as np
import ccxt # The library for connecting to exchanges


# --- CONFIGURATION ---
# This section holds all the parameters for the strategy.
# In a real-world system, this would be loaded from a secure config file.
CONFIG = {
   # --- Exchange & API ---
   # IMPORTANT: Add your API keys here
   "exchange": "bybit", # CCXT exchange ID
   "api_key": "YOUR_API_KEY",
   "api_secret": "YOUR_API_SECRET",
  
   # --- Strategy Parameters ---
   "symbol": "K/USDT",
   "timeframe": "1m",
   "capital_base": 5000, # Starting capital in USDT
   "risk_per_trade_percent": 1.0, # Risk 1% of total capital per trade
   "take_profit_1_percent": 1.5,
   "take_profit_2_percent": 2.5,
   "profit_lock_stop_loss_percent": 0.5,
   "fast_ma_period": 5,   # SMA
   "slow_ma_period": 10,  # EMA
   "volatility_ma_period": 50, # EMA for volatility filter
   "volatility_threshold_percent": 0.1 # Min % distance between slow MA and volatility MA
}


# --- LOGGING SETUP ---
# Professional systems require detailed logging to track decisions and errors.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(agent)s - %(levelname)s - %(message)s')


def get_logger(agent_name):
   """Creates a logger with a specific agent name."""
   return logging.LoggerAdapter(logging.getLogger(), {'agent': agent_name})


# --- AGENT 1: DATA HANDLER ---
# This agent's sole responsibility is to provide clean market data from the live exchange.
class DataHandler:
   def __init__(self, exchange):
       self.logger = get_logger(self.__class__.__name__)
       self.exchange = exchange
       self.logger.info(f"Initialized for {self.exchange.id} exchange.")


   def get_latest_bars(self, symbol, timeframe, n_bars=100):
       """Fetches the latest N bars from the exchange."""
       try:
           # CCXT fetches data in the format: [timestamp, open, high, low, close, volume]
           ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=n_bars)
           df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
           df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
           df.set_index('timestamp', inplace=True)
           return df
       except Exception as e:
           self.logger.error(f"Error fetching data from exchange: {e}")
           return None


# --- AGENT 2: STRATEGY & SIGNAL GENERATOR ---
# This agent is the brain. It performs calculations and generates signals.
class Strategy:
   def __init__(self, config):
       self.logger = get_logger(self.__class__.__name__)
       self.config = config
       self.logger.info("Strategy agent initialized with provided configuration.")


   def _calculate_indicators(self, data):
       """Calculates all necessary technical indicators."""
       df = data.copy()
       df['sma_fast'] = df['close'].rolling(window=self.config['fast_ma_period']).mean()
       df['ema_slow'] = df['close'].ewm(span=self.config['slow_ma_period'], adjust=False).mean()
       df['ema_volatility'] = df['close'].ewm(span=self.config['volatility_ma_period'], adjust=False).mean()
       return df


   def generate_signal(self, data):
       """
       Analyzes the data and generates a buy signal or a None.
       Returns a signal type ('BUY', 'EXIT') or None.
       """
       if len(data) < self.config['volatility_ma_period']:
           self.logger.warning("Not enough data to generate signals yet.")
           return None


       df = self._calculate_indicators(data)
      
       last_row = df.iloc[-1]
       prev_row = df.iloc[-2]


       volatility_spread = abs(last_row['ema_slow'] - last_row['ema_volatility']) / last_row['ema_slow'] * 100
       if volatility_spread < self.config['volatility_threshold_percent']:
           self.logger.info(f"Market too flat (spread: {volatility_spread:.2f}%). Filtering out signals.")
           return None


       buy_signal = (prev_row['ema_slow'] <= prev_row['sma_fast']) and (last_row['ema_slow'] > last_row['sma_fast'])
       exit_signal = (prev_row['ema_slow'] >= prev_row['sma_fast']) and (last_row['ema_slow'] < last_row['sma_fast'])


       if buy_signal:
           self.logger.info(f"BUY SIGNAL generated at price {last_row['close']:.5f}")
           return {'type': 'BUY', 'price': last_row['close']}
      
       if exit_signal:
           self.logger.info(f"EXIT SIGNAL generated at price {last_row['close']:.5f}")
           return {'type': 'EXIT', 'price': last_row['close']}


       return None


# --- AGENT 3: PORTFOLIO & RISK MANAGER ---
# This agent manages our capital, positions, and risk rules.
class PortfolioManager:
   def __init__(self, config, executor):
       self.logger = get_logger(self.__class__.__name__)
       self.config = config
       self.executor = executor
       self.cash = self.config['capital_base']
       self.position = None
       self.logger.info(f"Portfolio initialized with cash: ${self.cash:,.2f}")


   def on_signal(self, signal):
       """Handles a signal from the Strategy agent."""
       if signal['type'] == 'BUY' and self.position is None:
           self.execute_buy(signal)
       elif signal['type'] == 'EXIT' and self.position is not None:
           self.execute_sell(signal, "MA Crossover Exit")


   def execute_buy(self, signal):
       """Calculates position size and instructs the executor to buy."""
       risk_amount = self.cash * (self.config['risk_per_trade_percent'] / 100)
       entry_price = signal['price']
       quantity = risk_amount / entry_price
      
       if self.cash < (quantity * entry_price):
           self.logger.error("Insufficient cash to execute trade.")
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
       self.cash -= quantity * entry_price # Assume order fills instantly for this simulation
       self.logger.info(f"EXECUTED BUY: {quantity:.2f} units of {self.config['symbol']} at ~${entry_price:.5f}")


   def execute_sell(self, signal, reason=""):
       """Instructs the executor to close the current position."""
       if not self.position: return
      
       exit_price = signal['price']
       quantity_to_sell = self.position['quantity']
      
       self.executor.create_market_sell_order(self.config['symbol'], quantity_to_sell)
      
       pnl = (exit_price - self.position['entry_price']) * quantity_to_sell
       self.cash += quantity_to_sell * exit_price # Assume order fills instantly
       self.logger.info(f"EXECUTED SELL ({reason}): Closed position at ${exit_price:.5f}. PnL: ${pnl:.2f}. New cash: ${self.cash:,.2f}")
       self.position = None


   def check_trade_management(self, current_price):
       """Manages take-profits and dynamic stops for an open position."""
       if not self.position: return


       if not self.position['tp1_hit'] and current_price >= self.position['tp1_price']:
           self.logger.info("TP1 HIT!")
           quantity_to_sell = self.position['quantity'] / 2
           self.executor.create_market_sell_order(self.config['symbol'], quantity_to_sell)
          
           pnl = (self.position['tp1_price'] - self.position['entry_price']) * quantity_to_sell
           self.cash += quantity_to_sell * self.position['tp1_price']
           self.position['quantity'] -= quantity_to_sell
           self.position['tp1_hit'] = True
           self.logger.info(f"Took partial profit of ${pnl:.2f}. Remaining quantity: {self.position['quantity']:.2f}.")
           self.logger.info(f"PROFIT-LOCK STOP LOSS ACTIVATED at ${self.position['profit_lock_sl_price']:.5f}")


       if self.position['tp1_hit'] and current_price <= self.position['profit_lock_sl_price']:
           self.execute_sell({'price': self.position['profit_lock_sl_price']}, "Profit-Lock Stop Loss")
           return


       if self.position['tp1_hit'] and current_price >= self.position['tp2_price']:
           self.execute_sell({'price': self.position['tp2_price']}, "TP2 Hit")
           return


# --- AGENT 4: EXECUTION HANDLER ---
# This agent interacts directly with the exchange to place orders.
class ExecutionHandler:
   def __init__(self, exchange):
       self.logger = get_logger(self.__class__.__name__)
       self.exchange = exchange
       self.logger.info("Execution agent initialized for live trading.")


   def create_market_buy_order(self, symbol, quantity):
       """Places a market buy order."""
       self.logger.info(f"Sending MARKET BUY order for {quantity:.4f} {symbol}.")
       try:
           # Uncomment the line below to run live
           # return self.exchange.create_market_buy_order(symbol, quantity)
           pass
       except Exception as e:
           self.logger.error(f"Error placing market buy order: {e}")


   def create_market_sell_order(self, symbol, quantity):
       """Places a market sell order."""
       self.logger.info(f"Sending MARKET SELL order for {quantity:.4f} {symbol}.")
       try:
           # Uncomment the line below to run live
           # return self.exchange.create_market_sell_order(symbol, quantity)
           pass
       except Exception as e:
           self.logger.error(f"Error placing market sell order: {e}")


# --- AGENT 5: ORCHESTRATOR ---
# This is the main loop that coordinates all the agents for live trading.
class Orchestrator:
   def __init__(self, config):
       self.logger = get_logger(self.__class__.__name__)
       self.config = config
      
       # Initialize the exchange with API keys
       exchange_class = getattr(ccxt, config['exchange'])
       self.exchange = exchange_class({
           'apiKey': config['api_key'],
           'secret': config['api_secret'],
       })
      
       self.data_handler = DataHandler(self.exchange)
       self.executor = ExecutionHandler(self.exchange)
       self.portfolio = PortfolioManager(config, self.executor)
       self.strategy = Strategy(config)
       self.logger.info("Orchestrator initialized for LIVE TRADING.")


   def run(self):
       """The main event loop for live trading."""
       self.logger.info("--- Starting Live Trading System ---")
       self.logger.warning("Ensure you have installed necessary libraries: pip install pandas numpy ccxt")
       self.logger.warning("Orders are currently commented out in ExecutionHandler for safety.")
      
       while True:
           try:
               latest_data = self.data_handler.get_latest_bars(self.config['symbol'], self.config['timeframe'])
               if latest_data is None or latest_data.empty:
                   self.logger.warning("Could not fetch data. Retrying in 60 seconds.")
                   time.sleep(60)
                   continue
              
               current_price = latest_data.iloc[-1]['close']


               if self.portfolio.position:
                   self.portfolio.check_trade_management(current_price)


               signal = self.strategy.generate_signal(latest_data)


               if signal:
                   self.portfolio.on_signal(signal)


               # Wait for the next 1-minute candle
               time.sleep(60)


           except KeyboardInterrupt:
               self.logger.info("System shutdown requested by user.")
               break
           except Exception as e:
               self.logger.error(f"An unexpected error occurred in the main loop: {e}")
               time.sleep(60) # Wait before retrying


if __name__ == "__main__":
   # Ensure API keys are set before running
   if CONFIG['api_key'] == "YOUR_API_KEY" or CONFIG['api_secret'] == "YOUR_API_SECRET":
       print("!!! CRITICAL ERROR: Please enter your API key and secret in the CONFIG section before running.")
   else:
       system = Orchestrator(CONFIG)
       system.run()