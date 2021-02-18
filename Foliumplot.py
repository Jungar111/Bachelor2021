import folium
from Data_cleaning import clean_paloalto
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize, rgb2hex

class foliumMapPlot():
    def __init__(self):
        self.c = clean_paloalto()
        self.data = self.c.clean_data()


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

    def create_HTML(self, name,ptime, pmean):
        HTML = f"<h3>Name: {name}</h3> \n <h4>Mean charge time:</h4> <p>{ptime} </p> \n <h4> Mean kWh charged:</h4> <p>{pmean}</p>"

        return HTML


    def foliumplot(self):
        lat, lon = self.getloc(self.data)
        ptime, pmean, PSTIME = self.createdf()
        col = [mdates.date2num(i) for i in PSTIME]
        cmap = cm.autumn
        norm = Normalize(vmin=min(col), vmax=max(col))
        m = folium.Map(location=[37.435, -122.16], tiles="Stamen Toner", zoom_start=13)
        
        for i in range(len(lat)):
            dat = self.data[(self.data["Latitude"] == lat[i]) & (self.data["Longitude"] == lon[i])]
            folium.Circle(
                radius=np.round(pmean[i],2)/5,
                location=[lat[i], lon[i]],
                popup=self.create_HTML(dat["Station Name"].unique()[0], np.round(ptime[i],2), np.round(pmean[i],2)),
                color=rgb2hex(cmap(norm(col[i]))),
                fill=True,
            ).add_to(m)


        m.save("test.html")

if __name__ == '__main__':
    p = foliumMapPlot()
    p.foliumplot()