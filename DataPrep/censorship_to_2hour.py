import pandas as pd
import numpy as np



def proportionalsplit(s, freq="2H"):
    '''
    From StackOverflow: https://stackoverflow.com/questions/66274081/how-to-discretize-time-series-with-overspilling-durations/66280942#66280942
    '''
    st = s["Start Date"]
    etCharge = st + pd.Timedelta(minutes=s["Charging Time (mins)"])
    trCharge = pd.date_range(st.floor(freq), etCharge, freq=freq)
    etPark = st + pd.Timedelta(minutes=s["Total Duration (mins)"])
    trPark = pd.date_range(st.floor(freq), etPark, freq=freq)
    lmin = {"2H":120}
    #print(etCharge)
    #print(trCharge)
    #print(etPark)
    #print(trPark)
    
    # ratio of how numeric values should be split across new buckets
    ratioCharge = np.minimum((np.where(trCharge<st, trCharge.shift()-st, etCharge-trCharge)/(10**9*60)).astype(int), np.full(len(trCharge),lmin[freq]))
    ratioCharge = ratioCharge / ratioCharge.sum()
    #print(ratioCharge)
    ratioPark = np.minimum((np.where(trPark<st, trPark.shift()-st, etPark-trPark)/(10**9*60)).astype(int), np.full(len(trPark),lmin[freq]))
    ratioPark = ratioCharge / ratioCharge.sum()
    #print(ratioPark)
    
    return {"Start Date":trCharge, "Original Charge Duration":np.full(len(trCharge), s["Charging Time (mins)"]), 
                "Original Park Duration":np.full(len(trCharge), s["Total Duration (mins)"]), 
                "Original Start":np.full(len(trCharge), s["Start Date"]), 
                "Charging Time (mins)": s["Charging Time (mins)"] * ratioCharge,
                "Parking Time (mins)": s["Total Duration (mins)"] * ratioPark,
                "Energy (kWh)": s["Energy (kWh)"] * ratioCharge,
                #"CenterLon": np.full(len(trCharge),s["CenterLon"]), "CenterLat": np.full(len(trCharge),s["CenterLat"]), 
                "Fee (USD)": s["Fee (USD)"] * ratioCharge, 
                "Label": s["Label"]
            }

df = pd.read_csv("data\createdDat\Censorship_scheem\data_for_2hour_censored.csv")
df["Start Date"] = pd.to_datetime(df["Start Date"])
df["End Date"] = pd.to_datetime(df["End Date"])

df_days = pd.concat([pd.DataFrame(v) for v in df.apply(proportionalsplit, axis=1).values]).reset_index(drop=True)
df_days.to_csv("data\createdDat\Censorship_scheem\hour_two_data.csv") 