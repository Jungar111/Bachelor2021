import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import plotly.express as px
import sys
sys.path.append("..")
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt
from collections import Counter



class Buckets:
    def __init__(self):
        self.df = pd.read_csv('data/createdDat/CenteredData.csv')
        self.df = self.to_date(self.df)
        self.df = self.to_float(self.df)
        


    def to_float(self,df):
        df["Charging Time (hh:mm:ss)"][df["Charging Time (hh:mm:ss)"]==" -   "]=0
        df["Charging Time (hh:mm:ss)"]=df["Charging Time (hh:mm:ss)"].dt.hour * 60 + df["Charging Time (hh:mm:ss)"].dt.minute + df["Charging Time (hh:mm:ss)"].dt.second/60

        df["Total Duration (hh:mm:ss)"][df["Total Duration (hh:mm:ss)"]==" -   "]=0
        df["Total Duration (hh:mm:ss)"]=df["Total Duration (hh:mm:ss)"].dt.hour *60 + df["Total Duration (hh:mm:ss)"].dt.minute + df["Total Duration (hh:mm:ss)"].dt.second/60
        return df

    def to_date(self,df):
        df["Start Date"]=pd.to_datetime(df["Start Date"],format="%Y-%m-%d %H:%M:%S",errors="coerce")
        df["End Date"]=pd.to_datetime(df["End Date"],format="%Y-%m-%d %H:%M:%S",errors="coerce")

        df["Total Duration (hh:mm:ss)"]=pd.to_datetime(df["Total Duration (hh:mm:ss)"],format="%Y-%m-%d %H:%M:%S")
        df["Charging Time (hh:mm:ss)"]=pd.to_datetime(df["Charging Time (hh:mm:ss)"],format="%Y-%m-%d %H:%M:%S")
        return df

    def proportionalsplit(self, s, freq="2H"):
        '''
        From StackOverflow: https://stackoverflow.com/questions/66274081/how-to-discretize-time-series-with-overspilling-durations/66280942#66280942
        '''
        st = s["Start Date"]
        etCharge = st + pd.Timedelta(minutes=s["Charging Time (hh:mm:ss)"])
        trCharge = pd.date_range(st.floor(freq), etCharge, freq=freq)
        etPark = st + pd.Timedelta(minutes=s["Total Duration (hh:mm:ss)"])
        trPark = pd.date_range(st.floor(freq), etPark, freq=freq)
        lmin = {"2H":120}
        # ratio of how numeric values should be split across new buckets
        ratioCharge = np.minimum((np.where(trCharge<st, trCharge.shift()-st, etCharge-trCharge)/(10**9*60)).astype(int), np.full(len(trCharge),lmin[freq]))
        ratioCharge = ratioCharge / ratioCharge.sum()

        ratioPark = np.minimum((np.where(trPark<st, trPark.shift()-st, etPark-trPark)/(10**9*60)).astype(int), np.full(len(trPark),lmin[freq]))
        ratioPark = ratioCharge / ratioCharge.sum()

        return {"Start Date":trCharge, "Original Duration":np.full(len(trCharge), s["Charging Time (hh:mm:ss)"]), 
                "Original Start":np.full(len(trCharge), s["Start Date"]), 
                "Original Index": np.full(len(trCharge), s.name),
                "Charging Time (hh:mm:ss)": s["Charging Time (hh:mm:ss)"] * ratioCharge,
                "Energy (kWh)": s["Energy (kWh)"] * ratioCharge,
                "Total Duration (hh:mm:ss)": s["Total Duration (hh:mm:ss)"] * ratioPark,
                "Longitude": np.full(len(trCharge),s["CenterLon"]), "Latitude": np.full(len(trCharge),s["Latitude"]), "Original Port Type": np.full(len(trCharge),s["Port Type"]), "Port Number": np.full(len(trCharge), s["Port Number"]), 
                "Fee": s["Fee"] * ratioCharge, "Label": s["Label"]
            }

    def countLevels(self, s):
        counts = Counter(s["Original Port Type"])
        s["Level 1"] = counts["Level 1"]
        s["Level 2"] = counts["Level 2"]

        return s

    def main(self):
        df2 = pd.concat([pd.DataFrame(v) for v in self.df.apply(self.proportionalsplit, axis=1).values]).reset_index(drop=True)
        # everything OK?

        # let's have a look at everything in 2H resample...
        df3 = df2.groupby(["Start Date","Label"], as_index = False).agg({**{c:lambda s: list(s) for c in df2.columns if "Original" in c},
                                        **{c:"sum" for c in ["Charging Time (hh:mm:ss)","Energy (kWh)", "Total Duration (hh:mm:ss)", "Port Number"]}})

        df3 = df3.drop(columns=["Original Duration", "Original Start", "Original Index"])
        df3 = df3.apply(self.countLevels, axis=1)

        return df3,df2


if __name__ == "__main__":
    b = Buckets()
    df3, df2 = b.main()
    df3.to_csv("TimeBuckets.csv")
    