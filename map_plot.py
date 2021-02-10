from numpy.lib.arraysetops import unique
from Palo_Alto_Initial import clean_paloalto
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import numpy as np
from MapBoxApi import MapBoxAPI
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable


c = clean_paloalto()
data =  c.clean_data()
print(data.columns)

BBox = (data.Longitude.min()- 0.001, data.Longitude.max() + 0.001, data.Latitude.min() - 0.001,   data.Latitude.max() + 0.001)


name = "PaloAlto.png"
api = MapBoxAPI()
api.get_image((BBox[0], BBox[1]), (BBox[2], BBox[3]), name, 500)


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


fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(data.Longitude, data.Latitude, zorder=1, alpha= 0.2, c=data["time"], s=data["mean"]/10)
ax.set_title('Plotting Spatial Data on Palo Alto Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

values = list(np.quantile(data["mean"]/10,[0.125, 0.5, 0.875]))
values = [np.round(i,2) for i in values]
l1 = ax.scatter([],[], s=values[0], edgecolors='none', c="grey")
l2 = ax.scatter([],[], s=values[1], edgecolors='none', c="grey")
l3 = ax.scatter([],[], s=values[2], edgecolors='none', c="grey")

labels = values
leg = ax.legend([l1, l2, l3], labels, frameon=True, fontsize=8,
handlelength=2, loc = 'lower left', borderpad = 1,
handletextpad=1, title='Mean no. charges', scatterpoints = 1)


ax.imshow(paloaltoimg, zorder=0, extent = BBox, aspect= 'equal')

cmap = plt.get_cmap("viridis")
norm = plt.Normalize(data["time"].min(), data["time"].max())
sm =  ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
cbar.ax.set_title("Avg. Time (mins)")

plt.savefig("SpacialPlot.png")

#plt.show()