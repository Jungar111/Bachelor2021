import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import plotly.express as px
import matplotlib.patches as mpatches

from Data_cleaning import clean_paloalto
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

        
        dfdate = pd.DataFrame(np.transpose([df1["Pairlocation"].unique(),first_use,last_use]))
        dfdate.columns =['Pairlocation',"First use by location", 'Last use by location']
        dfdate["Online time"] =  dfdate["First use by location"]-dfdate["Last use by location"]
        #dfdate = dfdate.sort_values(by = "Online time")
        

        # df2 = df.sort_values(by="MAC Address")

        # first_use2 = []
        # for first in df2["MAC Address"].unique():
        #    dates = (df2["Start Date"][df2["MAC Address"]==first])
        #    first_use2.append(dates.min())
        # last_use2 = []
        # for last in df2["MAC Address"].unique():
        #    dates = (df2["Start Date"][df2["MAC Address"]==last])
        #    last_use2.append(dates.max())


        
        # dfdate2 = pd.DataFrame(np.transpose([df1["MAC Address"].unique(),first_use2,last_use2]))
        # dfdate2.columns =['MAC Address',"First use by MAC", 'Last use by MAC']
        # dfdate2["Online time"] =  dfdate2["Last use by MAC"]-dfdate2["First use by MAC"]


        dfclean = df

        dfclean = dfclean[['Pairlocation', 'MAC Address']] 
        test2 = dfclean.drop_duplicates()

        dfmerge = dfdate.merge(test2, how='outer', on = 'Pairlocation')
        # dfmerge2 = dfdate2.merge(test2, how='outer', on = 'MAC Address')
        
        

        fig1 = px.timeline(dfmerge, x_start="First use by location", x_end="Last use by location", y="MAC Address", color="Pairlocation")
        fig1.show()
        # fig2 = px.timeline(dfmerge2, x_start="First use by MAC", x_end="Last use by MAC", y="Pairlocation", color="MAC Address")
        # fig2.show()

        ##### Works but takes a long time, see Chargings_pr_day.png instead in Teams 
        #chargings = []
        #for days in df["Start Date"].dt.date.unique():
        #    chargings.append(len(df["Charge Duration (mins)"][(df["Start Date"].dt.date==days) & (df["Charge Duration (mins)"]!=0)]))
        # 
        #plt.bar(df["Start Date"].dt.date.unique(),chargings,align='center',width=1.0)
        #plt.plot(df["Start Date"],df["Fee"], c="red")
        #plt.title("Number of chargings pr. day")
        #plt.ylabel("Chargings pr. day/Fee")
        #plt.xlabel("Dates")
        #red_patch = mpatches.Patch(color='blue', label='Number of chargings')
        #blue_patch = mpatches.Patch(color='red', label='Fee')
        #plt.legend(handles=[red_patch, blue_patch])
        #plt.show()


        # df["Start Date"][df["Start Date"].dt.year==2016] insert for given 
        # Clearly see weekend lows 

if __name__=='__main__':
    c = clean_paloalto()
    data = c.clean_data()
    v = viz()
    v.by_dateplot(data)
    
    