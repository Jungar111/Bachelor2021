import sys
sys.path.append("..")
import platform
import pandas as pd
from DataPrep.DataBuckets import Buckets
from DataPrep.LagCreation import lags
from sklearn import preprocessing
<<<<<<< HEAD
from geopy import distance
=======
import numpy as np
>>>>>>> 6d5cda2bb11d79a54166fdf0351e24d60db0b5fb

class importer:
    def __init__(self):
        if platform.system() == "Darwin":
            self.df = pd.read_csv("data/createdDat/TimeBuckets.csv")
            self.POIs = pd.read_csv("data/createdDat/points_of_int.csv")

        elif platform.system() == "Windows":
            self.df = pd.read_csv("data\\createdDat\\TimeBuckets.csv")
            self.POIs = pd.read_csv("data\\createdDat\\points_of_int.csv")


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
        self.df = self.POIs_within_radius(self.df, self.POIs, 500)
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

    def distance_calc (self, row, label): #inspired by https://stackoverflow.com/questions/44446862/calculate-distance-between-latitude-and-longitude-in-dataframe
        POIlocation = (row['Latitude'], row['Longitude'])
        LabelX = (row['Label'+ str(label) +'_Lat'], row['Label'+ str(label) +'_Lon'])

        return distance.distance(POIlocation, LabelX).meters
        
    def POIs_within_radius(self, df, poi_df, radius):
        df_unique_label = df.groupby('Label', group_keys=False).apply(lambda df: df.sample(1))
        df_unique_label = df_unique_label[['Label','CenterLat', 'CenterLon']]
        df_unique_label = df_unique_label.set_index('Label')
        
        for i in range(0,len(df_unique_label)): 
            df_poi['Label' + str(i) + '_Lat'] = df_unique_label.at[i,'CenterLat']
            df_poi['Label' + str(i) + '_Lon'] = df_unique_label.at[i,'CenterLon']
        
        for j in range(0,len(df_unique_label)): 
            df_poi[str(j) + '_Distance'] = df_poi.apply (lambda row: self.distance_calc (row, j),axis=1)

        fill = pd.DataFrame()
        for k in range(0,8):
            m = radius 
            LabelY = pd.DataFrame(df_poi[df_poi[(str(k) + '_Distance')] < m]['Category'].value_counts())
            LabelY = LabelY.rename(columns={"Category": k})
            LabelY = LabelY.T
            fill = empt.append(LabelY)
        fill = empt.reset_index()
        fill = empt.rename(columns={"index": "Label"})
        category_names = list(fill.columns[1:])
        category_names_count = ['# ' + x for x in category_names]
        fill = fill.rename(columns = dict(zip(category_names, category_names_count)))
        
        result = df.merge(fill, on = "Label")
        
        return result

if __name__ == "__main__":
    i = importer()
    df = i.LagCreation()
    print(df.head())
    print(df.columns)
    

    
    