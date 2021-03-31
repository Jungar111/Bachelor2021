import sys
sys.path.append(".")
import platform
import pandas as pd
from DataPrep.LagCreation import lags
from sklearn import preprocessing
from geopy import distance
import numpy as np
from pathlib import Path
import holidays


class importer:
    def __init__(self):
        pdf = str(Path("data", "createdDat", "TimeBuckets.csv").absolute())
        ppoi = str(Path("data", "createdDat", "points_of_int.csv").absolute())

        self.df = pd.read_csv(pdf)
        self.POIs = pd.read_csv(ppoi)

    def to_date(self,df):
        df["Start Date"]=pd.to_datetime(df["Start Date"],format="%Y-%m-%d %H:%M:%S",errors="coerce")
        return df

    def Import(self):
        self.df = self.to_date(self.df)
        self.df = self.df[self.df["Start Date"].dt.year < 2020]
        self.df = self.df.drop(columns=["Unnamed: 0","Original Port Type"])
        #rint(self.df.columns)
        self.df.columns = ['Start Date', 'Label', 'Charging Time (mins)', 'Energy (kWh)', 'Total Duration (mins)', 'Port Number','CenterLon', 'CenterLat', 'Level 1', 'Level 2']
        
        self.df = self.resampling()
        self.df = self.df.dropna()
        self.df = self.df.apply(self.standardizeConsumption, axis=1)
        #self.normalizedata()
        self.df = self.POIs_within_radius(self.df, self.POIs, 500)
        self.OneHotEncode()
        self.is_holiday()
        self.is_weekend()
        return self.df


    def resampling(self):
        labels = self.df["Label"].unique()

        dfClean2 = pd.DataFrame(columns = self.df.columns[1:])


        for label in labels:
            d = self.df[self.df["Label"] == label].resample("D", on="Start Date").agg({'Charging Time (mins)':'sum', 'Energy (kWh)':'sum', 'Total Duration (mins)':'sum', 'Port Number':'sum', 'CenterLon':'min', 'CenterLat':'min','Level 1':'sum', 'Level 2': 'sum'})
            d["Label"] = label
            
            dfClean2 = dfClean2.append(d)
        
        dfClean2 = dfClean2.reset_index()
        dfClean2 = dfClean2.rename(columns = {"index":"Start Date"})

        return dfClean2

    def standardizeConsumption(self, s):
        s["Energy (kWh)"] = s["Energy (kWh)"]/s["Port Number"]
        return s

    def LagCreation(self):
        l = lags()
        data = self.Import()

        lagsData = l.buildLaggedFeatures(data, ["Energy (kWh)"])
        return lagsData

    def is_holiday(self, df): 
        us_ca_holidays = holidays.CountryHoliday('US', state='CA')
        self.df['is_holiday'] = [1 if str(val).split()[0] in us_ca_holidays else 0 for val in self.df['Start Date']]
        #return df

    def is_holiday_control(self, df):
        us_ca_holidays = holidays.CountryHoliday('US', state='CA')
        dates = []
        holis = []
        for date in df[df['is_holiday'] == 1]['Start Date']: 
            datestr = str(date).split()[0]
            dates.append(datestr)

        for date in dates: 
            holi = us_ca_holidays.get(date)
            holis.append(holi)

        control = dict(zip(dates, holis))
    
        return control

    def is_weekend(self, df):
        self.df['is_weekend'] = (self.df['Start Date'].dt.weekday > 4).astype(int) 
        #return df
    
    def normalizedata(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        cols = self.df[["Start Date","Longitude","Latitude","Port Number","ClusterID"]]
        df_scaled = pd.DataFrame(min_max_scaler.fit_transform(self.df.drop(columns=["Start Date"])),columns=self.df.drop(columns=["Start Date"]).columns.T,index=self.df.index.T)
        self.df = df_scaled
        self.df[["Start Date","Longitude","Latitude","Port Number","ClusterID"]] = cols

    def OneHotEncode(self):
        cluster_dummy = pd.get_dummies(self.df.Label, prefix="Cluster")
        day_month_dummy = pd.get_dummies(self.df["Start Date"].dt.day, prefix="Month_Day")
        day_week_dummy = pd.get_dummies(self.df["Start Date"].dt.dayofweek, prefix="Week_Day")
        month_year_dummy = pd.get_dummies(self.df["Start Date"].dt.month, prefix="Year_Month")
        res = pd.concat([cluster_dummy,day_month_dummy,day_week_dummy,month_year_dummy], axis=1)
        self.df = pd.concat([self.df, res], axis=1)

    def distance_calc (self, row, label): #inspired by https://stackoverflow.com/questions/44446862/calculate-distance-between-latitude-and-longitude-in-dataframe
        POIlocation = (row['Latitude'], row['Longitude'])
        LabelX = (row['Label'+ str(label) +'_Lat'], row['Label'+ str(label) +'_Lon'])

        return distance.distance(POIlocation, LabelX).meters
        
    def POIs_within_radius(self, df, df_poi, radius):
        df_unique_label = df.groupby('Label', group_keys=False).apply(lambda df: df.sample(1))
        df_unique_label = df_unique_label[['Label','CenterLat', 'CenterLon']]
        df_unique_label = df_unique_label.set_index('Label')
        
        for i in range(len(df_unique_label)): 
            df_poi['Label' + str(i) + '_Lat'] = df_unique_label.at[i,'CenterLat']
            df_poi['Label' + str(i) + '_Lon'] = df_unique_label.at[i,'CenterLon']
        
        for j in range(len(df_unique_label)): 
            df_poi[str(j) + '_Distance'] = df_poi.apply (lambda row: self.distance_calc (row, j),axis=1)

        fill = pd.DataFrame()
        for k in range(len(df_unique_label)):
            m = radius 
            LabelY = pd.DataFrame(df_poi[df_poi[(str(k) + '_Distance')] < m]['Category'].value_counts())
            LabelY = LabelY.rename(columns={"Category": k})
            LabelY = LabelY.T
            fill = fill.append(LabelY)
        fill = fill.reset_index()
        fill = fill.rename(columns={"index": "Label"})
        category_names = list(fill.columns[1:])
        category_names_count = ['# ' + x for x in category_names]
        fill = fill.rename(columns = dict(zip(category_names, category_names_count)))
        
        result = df.merge(fill, on = "Label")
        # result = result.drop(columns = [# Event])

        return result

    
if __name__ == "__main__":
    i = importer()
    df = i.LagCreation()
    print(df.head())
    print(df.columns)
    

    
    