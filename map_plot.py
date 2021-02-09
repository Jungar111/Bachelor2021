from numpy.lib.arraysetops import unique
from Palo_Alto_Initial import clean_paloalto
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import numpy as np
from MapBoxApi import MapBoxAPI

c = clean_paloalto()
data =  c.clean_data()
print(data.columns)

BBox = (data.Latitude.min() - 0.001,   data.Latitude.max() + 0.001,      
         data.Longitude.min()- 0.001, data.Longitude.max() + 0.001)


name = "PaloAlto.png"
api = MapBoxAPI()
api.get_image((BBox[0], BBox[1]), (BBox[2], BBox[3]), name)


paloaltoimg = plt.imread(f'img/{name}')

ID = []
Date = []
Amount = []
Time = []

for date in data["Start Date"].dt.month.unique():
    print(date)
    for charger in data["Pairlocation"].unique():
        amount = data[(data["Start Date"].dt.month == date) & (data["Pairlocation"] == charger)].shape[0]
        time = data["Charge Duration (mins)"][(data["Start Date"].dt.month == date) & (data["Pairlocation"] == charger)].mean()
        ID.append(charger)
        Date.append(date)
        Amount.append(amount)
        Time.append(time)
        

df = pd.DataFrame({
    "ID":ID,
    "Date":Date,
    "Amount":Amount,
    "Time" : Time
})
print(df.head())
print(df.describe())
df["mean"] = 0
df["time"] = 0

for ID in df["ID"].unique():
    df["mean"][df["ID"] == ID] = df["Amount"][df["ID"] == ID].mean()
    df["time"][df["ID"] == ID] = df["Time"][df["ID"] == ID].mean()

data = data.merge(df[["ID","mean","time"]], left_on="Pairlocation", right_on ="ID")

data["Pairlocation"].unique()


print(df.head(10))
print(df.describe())
print(data.head(10))
print(data.Latitude.min() - 0.001,   data.Latitude.max() + 0.001,      
         data.Longitude.min()- 0.001, data.Longitude.max() + 0.001)

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(data.Latitude, data.Longitude, zorder=1, alpha= 0.2, c=data["time"], s=data["mean"]/10)
ax.set_title('Plotting Spatial Data on Palo Alto Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(paloaltoimg, zorder=0, extent = BBox, aspect= 'equal')


plt.show()