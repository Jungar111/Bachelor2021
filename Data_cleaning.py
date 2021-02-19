import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import platform
import scipy
import math
import plotly.express as px
import platform
from datetime import timedelta
class clean_paloalto:
    def __init__(self):
        if platform.system() == "Darwin":
            self.pa_data = pd.read_csv("data/ChargePoint Data CY20Q4.csv")
        elif platform.system() == "Windows":
            self.pa_data = pd.read_csv("data\ChargePoint Data CY20Q4.csv")
    
    def clean_data(self):
        # We can drop EVSE ID, since mac address has more obs.
        # "Address 2", 
        self.data=self.pa_data.drop(["EVSE ID","County","System S/N","Model Number","Transaction Date (Pacific Time)"],axis=1)
        self.to_date(self.data)
        self.data["Latitude"]=self.data["Latitude"].round(3)
        self.data["Longitude"]=self.data["Longitude"].round(3)
        self.data["MAC Address"]=self.data["MAC Address"].str.replace(":", "")
        self.valuta(self.data)
        #self.to_float(self.data)
        self.data=self.data.dropna()
        self.data.index=range(len(self.data))
        codes,uniques = pd.factorize(self.data["MAC Address"])
        self.data["ID"] = codes
        self.pair(self.data)
        self.locationdf(self.data)
        return self.data

    def valuta(self,df):
        # EUR = 1.1761
        # MXN = 0.047122
        df["Fee"][df["Currency"]=="EUR"]=df["Fee"][df["Currency"]=="EUR"]*1.1761
        df["Fee"][df["Currency"]=="MXN"]=df["Fee"][df["Currency"]=="MXN"]*0.047122
    # def to_float(self,df):
    #     df["Charge Duration (mins)"][df["Charge Duration (mins)"]==" -   "]=0
    #     df["Charge Duration (mins)"]=pd.to_numeric(df["Charge Duration (mins)"],errors='coerce')


    def to_date(self,df):
        df["Start Date"]=pd.to_datetime(df["Start Date"],format="%m/%d/%Y %H:%M", errors="coerce")
        df["End Date"]=pd.to_datetime(df["End Date"],format="%m/%d/%Y %H:%M", errors="coerce")
        df["Total Duration (hh:mm:ss)"]=pd.to_datetime(df["Total Duration (hh:mm:ss)"],format="%H:%M:%S", errors="coerce")
        df["Charging Time (hh:mm:ss)"]=pd.to_datetime(df["Charging Time (hh:mm:ss)"],format="%H:%M:%S", errors="coerce")

        # Fixing different time zones
        df["Start Date"][df["Start Time Zone"]=="UTC"]=df["Start Date"][df["Start Time Zone"]=="UTC"]+timedelta(hours=-8)
        df["End Date"][df["End Time Zone"]=="UTC"]=df["End Date"][df["End Time Zone"]=="UTC"]+timedelta(hours=-8)
        

    def pair(self,df, print=False):
        pairlocation = []
        for index, longitude in enumerate(df["Longitude"]):
            pairlocation.append(str(df["Latitude"][index])+"x"+str(longitude))
        df["Pairlocation"] = pairlocation

        emptymac = [[] for i in range(len(df["Pairlocation"].unique()))]
        location = dict(zip(df["Pairlocation"].unique(),emptymac))
        for pair in df["Pairlocation"].unique():
            location[pair]=list(df["MAC Address"][df["Pairlocation"]==pair].unique())

        if print:
            for key, value in location.items():
                if len(value)>1:
                    print(key,value)
    
    def locationdf(self,df):
        df1 = df.sort_values(by="Pairlocation")
        first_use = []
        for first in df1["Pairlocation"].unique():
            dates = (df1["Start Date"][df1["Pairlocation"]==first])
            first_use.append(dates.min())
        last_use = []
        for last in df1["Pairlocation"].unique():
            dates = (df1["Start Date"][df1["Pairlocation"]==last])
            last_use.append(dates.max())
        
        dfloc=pd.DataFrame()
        dfloc["Pairlocation"]=df1["Pairlocation"].unique()
        dfloc["First use"] = first_use
        dfloc["Last use"] = last_use

        loclat = []
        loclon = []
        for loc in df1["Pairlocation"].unique():
            lat = df1["Latitude"][df1["Pairlocation"]==loc].unique()
            lon = df1["Longitude"][df1["Pairlocation"]==loc].unique()
            loclat.append(lat[0])
            loclon.append(lon[0])

        dfloc["Latitude"] = loclat
        dfloc["Longitude"] = loclon
    

if __name__=='__main__':
    c = clean_paloalto()
    data = c.clean_data()

    print(c.grid(data))