import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from sys import platform

class clean_paloalto:
    def __init__(self):
        if platform == "win32":
            self.pa_data = pd.read_csv("data\ChargePoint Data 2017Q4.csv")
        elif platform == "darwin":
            self.pa_data = pd.read_csv("data/ChargePoint Data 2017Q4.csv")
        
    
    def clean_data(self):
        # We can drop EVSE ID, since mac address has more obs. 
        self.data=self.pa_data.drop(["Address 2","EVSE ID","County","System S/N","Model Number","Transaction Date (Pacific Time)"],axis=1)
        self.data=self.data.dropna()
        self.data.index=range(len(self.data))
        self.to_date(self.data)
        #self.to_float(self.data)
        #self.drop_strangedata(self.data)
        return self.data
    
    def drop_strangedata(self,df):
        df["Charge Duration (mins)"]=df[df["Charge Duration (mins)"]!=" 1,104 "]
    
    def to_float(self,df):
        df["Charge Duration (mins)"][df["Charge Duration (mins)"]==" -   "]=0
        df["Charge Duration (mins)"]=pd.to_numeric(df["Charge Duration (mins)"])
    
    def to_date(self,df):
        df["Start Date"]=pd.to_datetime(df["Start Date"],format="%m/%d/%Y %H:%M", errors="coerce")
        df["End Date"]=pd.to_datetime(df["End Date"],format="%m/%d/%Y %H:%M", errors="coerce")
        df["Total Duration (hh:mm:ss)"]=pd.to_datetime(df["Total Duration (hh:mm:ss)"],format="%H:%M:%S")
        df["Charging Time (hh:mm:ss)"]=pd.to_datetime(df["Charging Time (hh:mm:ss)"],format="%H:%M:%S")
        

class viz:
    def normdistr(self,data):
        columns = ["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Fee"]
        for col in columns:
            mean, var  = scipy.stats.distributions.norm.fit(data[col])
            x = np.linspace(np.min(data[col]),np.max(data[col]),100)
            fitted_data = scipy.stats.distributions.norm.pdf(x, mean, var)

            plt.hist(data[col], density=True)
            plt.plot(x,fitted_data,'r-')
            plt.show()
    
    def expdistr(self, data):
        columns = ["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Fee"]
        for col in columns:
            mean, var  = scipy.stats.distributions.expon.fit(data[col])
            x = np.linspace(0,mean+np.sqrt(var)*6,100)
            fitted_data = scipy.stats.distributions.expon.pdf(x, mean, var)

            plt.hist(data[col], density=True, bins=100)
            plt.plot(x,fitted_data,'r-')
            plt.show()
    
    def lognormdistr(self,data):
        columns = ["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Fee"]
        for col in columns:
            plt.hist(data[col], density=True, bins=100)
            sns.distplot(data[col], fit=scipy.stats.lognorm)
            plt.show()

    def pairsplot(self,data):
        sns.pairplot(data[["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Port Number","Postal Code","Fee"]])
        plt.show()

    def basisplots(self,data):
        fig, ax = plt.subplots(2,2)
        ax[0,0].scatter(data["MAC Address"],data["Energy (kWh)"],s=0.1) 
        ax[0,1].scatter(data["MAC Address"],data["Charge Duration (mins)"],s=0.1) 
        ax[1,0].scatter(data["MAC Address"],data["Gasoline Savings (gallons)"],s=0.1) 
        ax[1,1].scatter(data["MAC Address"],data["Fee"],s=0.1) 
        plt.show()

        plt.scatter(data["Start Date"],data["Energy (kWh)"],s=0.1)
        plt.show()


if __name__=='__main__':
    c = clean_paloalto()
    v = viz()
    data = c.clean_data()
    v.basisplots(data)
    #print(data["Energy (kWh)"].dtypes)
    #c.pairsplot()
    #c.expdistr()
    #v.lognormdistr(data)

