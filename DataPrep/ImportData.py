import sys
sys.path.append(".")
import platform
import pandas as pd
from DataPrep.DataBuckets import Buckets

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
        self.df = self.df.drop(columns=["Original Duration", "Original Start", "Unnamed: 0", "Original Index","Original Port Type"])
        self.df.columns = ["Start Date", "Charging Time (mins)", "Energy (kWh)", "Total Duration (mins)", "Longitude", "Latitude", "Port Number", "Fee", "ClusterID"]
        self.df=self.df.dropna()
        return self.df


if __name__ == "__main__":
    i = importer()
    df = i.Import()
    
    

    
    