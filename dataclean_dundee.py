import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import os

"""
A class for cleaning the dundee data
"""

class clean_dundee:
    def __init__(self):
        self.datapaths = ["data/Dundee/cp-data-dec-mar-2018.csv", "data/Dundee/cp-data-mar-may-2018.csv", "data/Dundee/cpdata.csv"]


    def to_dt(self, df):
        df["Start Date"] = pd.to_datetime(df["Start Date"]).dt.date
        df["End Date"] = pd.to_datetime(df["End Date"]).dt.date
        df["Start Time"] = pd.to_datetime(df["Start Time"]).dt.time
        df["End Time"] = pd.to_datetime(df["End Time"]).dt.time

    def merge_data(self):
        dec_mar_dat = pd.read_csv(self.datapaths[0])
        mar_may_dat = pd.read_csv(self.datapaths[1])
        cp_dat = pd.read_csv(self.datapaths[2])

        dundee = cp_dat.append(dec_mar_dat.append(mar_may_dat))
        return dundee
    
    def clean_data(self):
        dundee = self.merge_data()
        dundee = dundee.dropna()
        self.to_dt(dundee)

        dundee["datetime_start"] = dundee.apply(lambda r : pd.datetime.combine(r['Start Date'],r['Start Time']),1)
        dundee["datetime_end"] = dundee.apply(lambda r : pd.datetime.combine(r['End Date'],r['End Time']),1)
        dundee["chargeTime"] = dundee.apply(lambda r : pd.Timedelta(r["datetime_end"]-r["datetime_start"]).seconds/3600,1)
        
        return dundee



if __name__ == '__main__':
    c = clean_dundee()
    dundee = c.clean_data()
    #dundee.to_csv("testdund.csv")
    
    

    

    