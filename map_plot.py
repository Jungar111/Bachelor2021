from numpy.lib.arraysetops import unique
from Data_cleaning import clean_paloalto
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import numpy as np
from MapBoxApi import MapBoxAPI
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates

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

def getloc(df):
        lat = []
        lon = []

        pairloc = list(df["Pairlocation"].unique())
        pairloc = [i.split("x") for i in pairloc]

        for pair in pairloc:
            lat.append(float(pair[0]))
            lon.append(float(pair[1]))
        
        return lat,lon

lat, lon = getloc(data)

ptime = []
pmean = []
pstime = []

for pair in data["Pairlocation"].unique():
    ptime.append(data["time"][data["Pairlocation"] == pair].unique().tolist())
    pmean.append(data["mean"][data["Pairlocation"] == pair].unique().tolist())
    dates = data["Start Date"][data["Pairlocation"] == pair].dt.date
    pstime.append(dates.unique())

print("HALLO")
print(type(data["Start Date"][0]))

ptime = flat_list = [item for sublist in ptime for item in sublist]

pmean = flat_list = [item/10 for sublist in pmean for item in sublist]

PSTIME = []
for pst in pstime:
    app = min(pst)
    print(app)
    PSTIME.append(app)

print(ptime)
data["Start Date"][data["Pairlocation"] == pair].unique().tolist()



fig, ax = plt.subplots(figsize = (13,7))
sc = ax.scatter(lon, lat, zorder=1, alpha= 0.6, c=[mdates.date2num(i) for i in PSTIME])
ax.set_title('Plotting Spatial/Time Data on Palo Alto Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

# values = list(np.quantile(data["mean"]/10,[0.125, 0.5, 0.875]))
# values = [np.round(i,2) for i in values]
# l1 = ax.scatter([],[], s=values[0], edgecolors='none', c="grey")
# l2 = ax.scatter([],[], s=values[1], edgecolors='none', c="grey")
# l3 = ax.scatter([],[], s=values[2], edgecolors='none', c="grey")

# labels = values
# leg = ax.legend([l1, l2, l3], labels, frameon=True, fontsize=8,
# handlelength=2, loc = 'lower left', borderpad = 1,
# handletextpad=1, title='Mean no. charges', scatterpoints = 1)


ax.imshow(paloaltoimg, zorder=0, extent = BBox, aspect= 'equal')


loc = mdates.AutoDateLocator()
cbar = fig.colorbar(sc, ticks=loc,
                 format=mdates.AutoDateFormatter(loc),
                 ax = ax)

cbar.ax.set_title("First use")

plt.savefig("TIMEPLOT.png")

#plt.show()