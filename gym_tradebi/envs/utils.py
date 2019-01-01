
from enum import Enum, auto
import matplotlib.dates as mdates
import matplotlib.finance as mf
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict

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

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def close(self, current_price, exit_time):

        profit = 0

        if(self.trade_type == TradeType.LONG):
            profit = (current_price - self.price) / self.precision * self.lot
        else:
            profit = (self.price - current_price) / self.precision * self.lot

        self.exit_time = exit_time

        return profit

class Portfolio:

    def __init__(self):

        #Current existing trades        
        self.trades = []

        #Cumulative reward in this run (in pips)
        self.total_reward = 0

        #Cumulative trades in this run
        self.total_trades = 0

        self.average_profit_per_trade = 0

        #History of cumulative reward 
        self.equity_curve = [] #TO BE OUTSOURCED TO AGENT
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Record:

    def __init__(self):

        self.portfolios = OrderedDict()

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def add(self, portfolio):

        print("Len(portfolios): " + str(len(self.portfolios)))
        
        self.portfolios[portfolio.total_reward] = portfolio

        #if portfolio.total_reward >= 17170:
        #    self.portfolios.insert(0,portfolio)

        '''if len(self.portfolios) == 0:
            #self.portfolios.insert(0,portfolio)
            self.portfolios.append(portfolio)
        else:
            print("Existing total_reward: " + str(self.portfolios[0].total_reward))
            print("New total_reward: " + str(portfolio.total_reward))
            if self.portfolios[0].total_reward < portfolio.total_reward:
                self.portfolios.insert(0,portfolio)
        '''


class Visual:

    def __init__(self):
        self.CHART_SIZE = (20,10)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def rewardPlot(self, record, best_models, TYPE, top_n=3):

        arr = np.asarray(record)
        
        #Return the index based on top n
        top_10_episodes = [x[0] for x in best_models]
        top_10_index = np.array(top_10_episodes) -1

        top_n_episodes = top_10_episodes[:top_n]
        top_n_index = top_10_index[:top_n]
        top_n_rewards = arr[top_n_index]


        fig = plt.figure(figsize=self.CHART_SIZE)
        ax = fig.add_subplot(111)
        color = 'b-' if TYPE=='Total' else 'r-'
        ax.plot(record, color)
        ax.set_title("%s Reward (Showing Top %s)"%(TYPE,top_n), fontdict={'fontsize':20})
        ax.set_xlabel("Episodes")
        

        textString = "TOP {}: \n".format(top_n)
        for i, r in enumerate(top_n_rewards):
            
            epi= top_n_episodes[i]

            textString += "Episode {}: {} \n".format(epi, record[epi-1])
        
        ax.text(0.75, 0.5, textString, fontsize=10, verticalalignment='top',transform=ax.transAxes,
        bbox={'alpha':0.5, 'pad':10})

        plt.show()

    def candelPlot(self, ohlc, portfolio):

        #Filter out buys and sells
        buys = [x for x in portfolio.trades if x.trade_type == TradeType.LONG]
        sells = [x for x in portfolio.trades if x.trade_type == TradeType.SHORT]

        print("BUY Len: " + str(len(buys)))
        print("Sell Len: " + str(len(sells)))

        #make OHLC ohlc matplotlib friendly
        datetime_index = mdates.date2num(ohlc.index.to_pydatetime())
        
        proper_feed = list(zip(
            datetime_index, 
            ohlc.Open.tolist(), 
            ohlc.High.tolist(), 
            ohlc.Low.tolist(), 
            ohlc.Close.tolist()
            ))

        #actual PLotting
        fig, (ax, ax2) = plt.subplots(2,1, figsize=self.CHART_SIZE)

        ax.set_title('Action History', fontdict={'fontsize':20})
        
        all_days= mdates.DayLocator()
        ax.xaxis.set_major_locator(all_days)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

        #Candlestick chart
        mf.candlestick_ohlc(
            ax,
            proper_feed,
            width=0.02,
            colorup='green',
            colordown='red'
        )

        #Buy indicator
    
        ax.plot(
            mdates.date2num([buy.entry_time for buy in buys]),
            [buy.price-0.001 for buy in buys],
            'b^',
            alpha=1.0
        )

        #Sell indicator
        ax.plot(
            mdates.date2num([sell.entry_time for sell in sells]),
            [sell.price+0.001 for sell in sells],
            'rv',
            alpha=1.0
        )



    def ohlcPlot(self, journal, ohlc, equity_curve, PRECISION=0.0001):

        #Filter out buys and sells
        buys = [x for x in journal if x['Type']=='BUY']
        sells = [x for x in journal if x['Type']=='SELL']

        #make OHLC ohlc matplotlib friendly
        datetime_index = mdates.date2num(ohlc.index.to_pydatetime())
        
        proper_feed = list(zip(
            datetime_index, 
            ohlc.Open.tolist(), 
            ohlc.High.tolist(), 
            ohlc.Low.tolist(), 
            ohlc.Close.tolist()
            ))

        #actual PLotting
        fig, (ax, ax2) = plt.subplots(2,1, figsize=self.CHART_SIZE)

        ax.set_title('Action History', fontdict={'fontsize':20})
        
        all_days= mdates.DayLocator()
        ax.xaxis.set_major_locator(all_days)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

        #Candlestick chart
        mf.candlestick_ohlc(
            ax,
            proper_feed,
            width=0.02,
            colorup='green',
            colordown='red'
        )


        #Buy indicator
    
        ax.plot(
            mdates.date2num([buy['Entry Time'] for buy in buys]),
            [buy['Entry Price']-0.001 for buy in buys],
            'b^',
            alpha=1.0
        )

        #Sell indicator
        ax.plot(
            mdates.date2num([sell['Entry Time'] for sell in sells]),
            [sell['Entry Price']+0.001 for sell in sells],
            'rv',
            alpha=1.0
        )


        #Secondary Plot
        ax2.set_title("Equity")
        ax2.plot(equity_curve)

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        plt.show()

