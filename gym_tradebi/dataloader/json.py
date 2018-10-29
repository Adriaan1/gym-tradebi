import numpy as np
import pandas as pd

import json
import datetime
import time

class Dataloader():

    def __init__(self, fileName):

        with open(fileName) as f:
            response = json.load(f)

        received = response["candles"]

        #Store the history in a dictionary of list
        data = {
            "Open" : [],
            "High" : [],
            "Low" : [],
            "Close" : [],
            "Volume": [],
            "Date_Time" : [],        
        }

        for r in received:
            data["Open"].append(float (r["mid"]["o"]) )
            data["High"].append(float (r["mid"]["h"]))
            data["Low"].append(float (r["mid"]["l"]))
            data["Close"].append(float (r["mid"]["c"]))
            data["Volume"].append(float(r["volume"]) )
            data["Date_Time"].append(r["time"])
            
       
        #Convert the dictionary to pandas DataFrame:
        #Date_Time (index) | Open | High | Low | Close | Volume
        self.df = pd.DataFrame(data)

        self.df["Date_Time"] = pd.to_datetime(self.df["Date_Time"])
        self.df.set_index(["Date_Time"], inplace=True)  

    def getData(self):

        return self.df      