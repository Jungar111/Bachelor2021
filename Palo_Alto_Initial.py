import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
class clean_paloalto:
    def __init__(self):
        self.pa_data = pd.read_csv("data\ChargePoint Data 2017Q4.csv")
    
    def clean_data(self):
        # We can drop EVSE ID, since mac address has more obs. 
        self.data=self.pa_data.drop(["Address 2","EVSE ID","County","System S/N","Model Number","Transaction Date (Pacific Time)"],axis=1)
        self.data=self.data.dropna()
        self.data.index=range(len(self.data))
        codes,uniques = pd.factorize(self.data["MAC Address"])
        self.data["ID"] = codes
        self.to_date(self.data)
        self.to_float(self.data)
        self.pair(self.data)
        return self.data

    def to_float(self,df):
        df["Charge Duration (mins)"][df["Charge Duration (mins)"]==" -   "]=0
        df["Charge Duration (mins)"]=pd.to_numeric(df["Charge Duration (mins)"],errors='coerce')


    def to_date(self,df):
        df["Start Date"]=pd.to_datetime(df["Start Date"],format="%m/%d/%Y %H:%M", errors="coerce")
        df["End Date"]=pd.to_datetime(df["End Date"],format="%m/%d/%Y %H:%M", errors="coerce")
        df["Total Duration (hh:mm:ss)"]=pd.to_datetime(df["Total Duration (hh:mm:ss)"],format="%H:%M:%S")
        df["Charging Time (hh:mm:ss)"]=pd.to_datetime(df["Charging Time (hh:mm:ss)"],format="%H:%M:%S")
    
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
        #sns.pairplot(data[["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Port Number","Postal Code","Fee"]])
        sns.pairplot(data[["Energy (kWh)","Charge Duration (mins)"]])
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

    def by_dateplot(self,df):
        df1 = df.sort_values(by="Pairlocation")
        first_use = []
        for first in df1["Pairlocation"].unique():
            dates = (df1["Start Date"][df1["Pairlocation"]==first])
            first_use.append(dates.min())
        last_use = []
        for last in df1["Pairlocation"].unique():
            dates = (df1["Start Date"][df1["Pairlocation"]==last])
            last_use.append(dates.max())
    
        plt.scatter(df["Pairlocation"].unique(),first_use)
        plt.scatter(df["Pairlocation"].unique(),last_use, c="red")
        plt.title("First and last charge for stations")
        plt.show()


        ##### Works but takes a long time, see Chargings_pr_day.png instead in Teams 
        #chargings = []
        #for days in df["Start Date"].dt.date.unique():
        #    chargings.append(len(df["Charge Duration (mins)"][(df["Start Date"].dt.date==days) & (df["Charge Duration (mins)"]!=0)]))
        
        #plt.bar(df["Start Date"].dt.date.unique(),chargings)
        #plt.title("Number of chargings pr. day")
        #plt.ylabel("Chargings pr. day")
        #plt.xlabel("Dates")
        #plt.show()



if __name__=='__main__':
    c = clean_paloalto()
    v = viz()
    data = c.clean_data()
    
    #v.basisplots(data)
    v.by_dateplot(data)
    #c.pair(data,true)


    #print(len(data["MAC Address"].unique()))
    #v.pairsplot(data)
    #v.expdistr(data)
    #v.lognormdistr(data)

