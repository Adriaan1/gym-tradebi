
from enum import Enum, auto

class TradeType(Enum):
     LONG = auto()
     SHORT = auto()

class Trade:

    def __init__(self, trade_type, lot, price, precision, entry_time):
        self.trade_type = trade_type
        self.lot = lot
        self.price = price
        self.precision = precision
        self.entry_time = entry_time 
        self.exit_time = None

    def close(self, current_price):

        profit = 0

        if(self.trade_type == TradeType.LONG):
            profit = (current_price - self.price) / self.precision * self.lot
        else:
            profit = (self.price - current_price) / self.precision * self.lot

        return profit

class Portfolio:

    def __init__(self):

        self.reset()
    
    def reset(self):

        #Current existing trades        
        self.trades = []

        #Cumulative reward in this run (in pips)
        self.total_reward = 0

        #Cumulative trades in this run
        self.total_trades = 0

        self.average_profit_per_trade = 0

        #History of cumulative reward 
        self.equity_curve = [] #TO BE OUTSOURCED TO AGENT
