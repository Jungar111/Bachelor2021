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
        self.data=self.pa_data.drop(["Address 2","EVSE ID","County","System S/N","Model Number"],axis=1)
        self.data=self.data.dropna()
        return self.data
    
    def normdistr(self):
        data = self.clean_data()
        columns = ["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Fee"]
        for col in columns:
            mean, var  = scipy.stats.distributions.norm.fit(data[col])
            x = np.linspace(np.min(data[col]),np.max(data[col]),100)
            fitted_data = scipy.stats.distributions.norm.pdf(x, mean, var)

            plt.hist(data[col], density=True)
            plt.plot(x,fitted_data,'r-')
            plt.show()
    
    def expdistr(self):
        data = self.clean_data()
        columns = ["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Fee"]
        for col in columns:
            mean, var  = scipy.stats.distributions.expon.fit(data[col])
            x = np.linspace(0,mean+np.sqrt(var)*6,100)
            fitted_data = scipy.stats.distributions.expon.pdf(x, mean, var)

            plt.hist(data[col], density=True, bins=100)
            plt.plot(x,fitted_data,'r-')
            plt.show()
    
    def lognormdistr(self):
        data = self.clean_data()
        columns = ["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Fee"]
        for col in columns:
            mean, var, skew  = scipy.stats.distributions.lognorm.fit(data[col])
            x = np.linspace(0.0000001,mean+np.sqrt(var)*6,100)
            fitted_data = scipy.stats.distributions.lognorm.pdf(x, mean, var, skew)

            plt.hist(data[col], density=True, bins=100)
            plt.plot(x,fitted_data,'r-')
            plt.show()

    def pairsplot(self):
        data = self.clean_data()
        sns.pairplot(data[["Energy (kWh)","GHG Savings (kg)","Gasoline Savings (gallons)","Port Number","Postal Code","Fee"]])
        plt.show()


if __name__=='__main__':
    c = clean_paloalto()
    data = c.clean_data()
    #c.pairsplot()
    #c.expdistr()
    c.lognormdistr()
    #print(math.log(data["Energy (kWh)"]))
    #print(data["Energy (kWh)"].describe())
    #print(data.apply(lambda r : math.exp(r["Energy (kWh)"]),1))
