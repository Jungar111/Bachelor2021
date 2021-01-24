import pandas as pd 
import matplotlib.pyplot as plt

dec_mar = pd.read_csv("data\Dundee\cp-data-dec-mar-2018.csv")
mar_may = pd.read_csv("data\Dundee\cp-data-mar-may-2018.csv")

data = dec_mar.append(mar_may)

print(data.head())
#print(data.isna().sum())
#print(data.shape)


plt.plot(data["Total kWh"])
plt.show()

