import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import plotly.express as px
from DataPrep.Data_cleaning import clean_paloalto

import numpy as np
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt

df = clean_paloalto().clean_data()

start = time.time()
'''
From StackOverflow: https://stackoverflow.com/questions/66274081/how-to-discretize-time-series-with-overspilling-durations/66280942#66280942
'''
def proportionalsplit(s, freq="2H"):
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
            "Total Duration (hh:mm:ss)": s["Total Duration (hh:mm:ss)"] * ratioPark
           }

df2 = pd.concat([pd.DataFrame(v) for v in df.apply(proportionalsplit, axis=1).values]).reset_index(drop=True)
# everything OK?


# let's have a look at everything in 2H resample...
df3 = df2.groupby(["Start Date"]).agg({**{c:lambda s: list(s) for c in df2.columns if "Original" in c},
                                **{c:"sum" for c in ["Charging Time (hh:mm:ss)","Energy (kWh)", "Total Duration (hh:mm:ss)"]}})



print(df2.head())
print(df3.head())

df3.to_csv("TimeBuckets.csv")

end = time.time()
print(end - start)