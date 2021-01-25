import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

dec_mar = pd.read_csv("data\Dundee\cp-data-dec-mar-2018.csv")
mar_may = pd.read_csv("data\Dundee\cp-data-mar-may-2018.csv")

data = dec_mar.append(mar_may)

# Fjerner na/missing values fra data 
#print(data.isna().sum())
data = data.dropna()

# Laver date om til datetime 
data["Start Date"] = pd.to_datetime(data["Start Date"])
data["End Date"] = pd.to_datetime(data["End Date"])

# De her virker ikke??

#data["Start Time"] = pd.to_datetime(data["Start Time"],format="%H:%M")
#data["End Time"] = pd.to_datetime(data["End Time"],format="%H:%M")

# Kører describe 
print(data.describe())
#cost kan fjernes 

# Finder det site der har negative værdier 
#print(data["Site"][data["Total kWh"]<0])

plt.scatter(data["Start Date"],data["Total kWh"])
plt.show()
#Der bliver ladet meget mere i starten af året end i slutningen??


# Finder corr mellem variable 
corr = data.corr()
print(corr)


# Mangler col med tid??
sns.pairplot(data)
plt.show()



