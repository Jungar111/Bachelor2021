import sys
sys.path.append("..")
import platform
import pandas as pd
from DataPrep.DataBuckets import Buckets
from DataPrep.LagCreation import lags
from sklearn import preprocessing
import numpy as np

class importer:
    def __init__(self):
        if platform.system() == "Darwin":
            self.df = pd.read_csv("data/createdDat/TimeBuckets.csv")
        elif platform.system() == "Windows":
            self.df = pd.read_csv("data\\createdDat\\TimeBuckets.csv")


    def to_date(self,df):
        df["Start Date"]=pd.to_datetime(df["Start Date"],format="%Y-%m-%d %H:%M:%S",errors="coerce")
        return df

    def Import(self):
        self.df = self.to_date(self.df)
        self.df = self.df[self.df["Start Date"].dt.year < 2020]
        self.df = self.df.drop(columns=["Unnamed: 0","Original Port Type"])
        #print(self.df.columns)
        self.df.columns = ['Start Date', 'ClusterID', 'Charging Time (mins)', 'Energy (kWh)', 'Total Duration (mins)', 'Port Number', 'Level 1', 'Level 2']
        self.df=self.df.dropna()
        
        self.df = self.df.apply(self.standardizeConsumption, axis=1)
        
        #self.normalizedata()
        self.OneHotEncode()
        return self.df

    def standardizeConsumption(self, s):
        s["Energy (kWh)"] = s["Energy (kWh)"]/s["Port Number"]
        return s

    def LagCreation(self):
        l = lags()
        data = self.Import()

        lagsData = l.buildLaggedFeatures(data, ["Energy (kWh)"])
        return lagsData
    
    def normalizedata(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        cols = self.df[["Start Date","Longitude","Latitude","Port Number","ClusterID"]]
        df_scaled = pd.DataFrame(min_max_scaler.fit_transform(self.df.drop(columns=["Start Date"])),columns=self.df.drop(columns=["Start Date"]).columns.T,index=self.df.index.T)
        self.df = df_scaled
        self.df[["Start Date","Longitude","Latitude","Port Number","ClusterID"]] = cols

    def OneHotEncode(self):
        cluster_dummy = pd.get_dummies(self.df.ClusterID, prefix="Cluster")
        day_month_dummy = pd.get_dummies(self.df["Start Date"].dt.day, prefix="Month_Day")
        day_week_dummy = pd.get_dummies(self.df["Start Date"].dt.dayofweek, prefix="Week_Day")
        month_year_dummy = pd.get_dummies(self.df["Start Date"].dt.month, prefix="Year_Month")
        res = pd.concat([cluster_dummy,day_month_dummy,day_week_dummy,month_year_dummy], axis=1)
        self.df = pd.concat([self.df, res], axis=1)
        


if __name__ == "__main__":
    i = importer()
    df = i.LagCreation()
    print(df.head())
    print(df.columns)
    

    
    