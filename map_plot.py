from Palo_Alto_Initial import clean_paloalto
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd

c = clean_paloalto()
data =  c.clean_data()
print(data.columns)

BBox = (data.Longitude.min() - 0.001,   data.Longitude.max() + 0.001,      
         data.Latitude.min()- 0.001, data.Latitude.max() + 0.001)

paloaltoimg = plt.imread('map.png')

ID = []
Date = []
Amount = []
for date in data["Start Date"].dt.date:
    for charger in data["ID"].unique():
        ID.append(charger)
        Date.append(date)
        Amount.append(data[data["Start Date"].dt.date == date & data["ID"] == ID].shape[1])

df = pd.DataFrame(list(zip(ID, Date, Amount)), columns=["ID","Date","Amount"])
print(df.head())

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(data.Longitude, data.Latitude, zorder=1, alpha= 0.2, c='b', s=0.1)
ax.set_title('Plotting Spatial Data on Palo Alto Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(paloaltoimg, zorder=0, extent = BBox, aspect= 'equal')


plt.show()