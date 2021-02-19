import folium
from Data_cleaning import clean_paloalto
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize, rgb2hex
import platform
from grid import gridmap

class foliumMapPlot():
    def __init__(self):
        self.c = clean_paloalto()
        self.data = self.c.clean_data()
        if platform.system() == "Darwin":
            self.POI = pd.read_csv("points_of_int1.csv")
        elif platform.system() == "Windows":
            self.POI = pd.read_csv("points_of_int1.csv")



    def createdf(self):
        ID = []
        Date = []
        Amount = []
        Time = []

        for date in self.data["Start Date"].dt.month.unique():
            print(date)
            for charger in self.data["Pairlocation"].unique():
                amount = self.data[(self.data["Start Date"].dt.month == date) & (self.data["Pairlocation"] == charger)].shape[0]
                time = self.data["Charging Time (hh:mm:ss)"][(self.data["Start Date"].dt.month == date) & (self.data["Pairlocation"] == charger)].mean()
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

        data = self.data.merge(df[["ID","mean","time"]], left_on="Pairlocation", right_on ="ID")

        ptime = []
        pmean = []
        pstime = []

        for pair in data["Pairlocation"].unique():
            ptime.append(data["time"][data["Pairlocation"] == pair].unique().tolist())
            pmean.append(data["mean"][data["Pairlocation"] == pair].unique().tolist())
            dates = data["Start Date"][data["Pairlocation"] == pair].dt.date
            pstime.append(dates.unique())

        ptime = [item for sublist in ptime for item in sublist]

        pmean =  [item/10 for sublist in pmean for item in sublist]

        PSTIME = []
        for pst in pstime:
            app = min(pst)
            print(app)
            PSTIME.append(app)


        return ptime, pmean, PSTIME

    def getloc(self,df):
            lat = []
            lon = []

            pairloc = list(df["Pairlocation"].unique())
            pairloc = [i.split("x") for i in pairloc]

            for pair in pairloc:
                lat.append(float(pair[0]))
                lon.append(float(pair[1]))
            
            return lat,lon

    def create_HTML(self, name,ptime, pmean, maddres, plug, mindate,maxdate):
        HTML = f"<h5>MAC Adress: {maddres}</h5> \n <h6>Name:</h6> <p>{name}</p> \n <h6>Mean charge time:</h6> <p>{ptime}</p> \n <h6> Mean kWh charged:</h6> <p>{pmean}</p>"
        HTML += f"\n <h6>Plug Type:</h6> <p>{plug}</p> \n <h6>First Use:</h6> <p>{mindate}</p> \n <h6>Last Use:</h6> <p>{maxdate}</p>"

        return HTML
    
    def create_POI_HTML(self, name, cat, subcat):
        HTML = f"<h5>Name: {name}</h5> \n <h6>Category:</h6> <p>{cat}</p> \n <h6>Subcategory:</h6> <p>{subcat}</p> \n"

        return HTML


    def foliumplot(self):
        m = folium.Map(location=[37.435, -122.16], tiles="Stamen Toner", zoom_start=13)
        groups = self.POI.Category.unique().tolist()
        colors = ['red','blue','gray','orange','beige','green','purple','pink','cadetblue','black']
        cols = dict(zip(groups,colors))

        for index, poi in self.POI.iterrows():
            folium.Circle(
                radius=15,
                location=[poi.Latitude, poi.Longitude],
                popup=self.create_POI_HTML(poi.Name, poi.Category, poi["Sub Category"]),
                color=cols[poi.Category],
                fill=False,
            ).add_to(m)

        lat, lon = self.getloc(self.data)
        ptime, pmean, PSTIME = self.createdf()
        col = [mdates.date2num(i) for i in PSTIME]
        cmap = cm.autumn
        norm = Normalize(vmin=min(col), vmax=max(col))
        
        for i in range(len(lat)):
            dat = self.data[(self.data["Latitude"] == lat[i]) & (self.data["Longitude"] == lon[i])]
            first_use = dat["Start Date"].min()
            last_use = dat["End Date"].min()
            folium.Circle(
                radius=np.round(pmean[i],2),
                location=[lat[i], lon[i]],
                popup=self.create_HTML(dat["Station Name"].unique()[0], ptime[i], np.round(pmean[i],2), dat["MAC Address"].unique()[0], dat["Plug Type"].unique()[0], first_use, last_use),
                color=rgb2hex(cmap(norm(col[i]))),
                fill=True,
            ).add_to(m)
            
        

        m.save("folium.html")

if __name__ == '__main__':
    p = foliumMapPlot()
    p.foliumplot()
    g = gridmap()
