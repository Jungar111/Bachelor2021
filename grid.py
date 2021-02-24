from Data_cleaning import clean_paloalto
from sklearn.cluster import KMeans
import pandas as pd
import folium
from Data_cleaning import clean_paloalto
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize, rgb2hex
import platform


class gridmap:
    def __init__(self):
        self.data = clean_paloalto().clean_data()

    def grid(self,c):
        
        kmeans = KMeans(n_clusters=c, random_state=0).fit(self.data[["Latitude","Longitude"]])
        clusters=kmeans.cluster_centers_
        label = kmeans.labels_
        
        griddf = self.data[["Start Date","Port Type","Latitude","Longitude","Fee","Pairlocation"]]
        griddf["Label"]=label
        # griddf.columns= ["Start Date","Port Type","Latitude","Longitude","Fee","Pairlocation","Label"]
        return griddf

class plot:
    def create_HTML(self,cluster):
        html=f"<h4> Cluster: {cluster}</h4>"
        return html

    def foliumplot(self):
        m = folium.Map(location=[37.435, -122.16], tiles="Stamen Toner", zoom_start=13)
        g = gridmap()
        grid = g.grid(8).groupby("Pairlocation")

        cmap = cm.viridis
        norm = Normalize(vmin=0, vmax=7)

        for index,c in grid:
            #print([c["Longitude"].unique()[0], c["Latitude"].unique()[0]])
            folium.Circle(
                radius=30,
                location=[c["Latitude"].unique()[0],c["Longitude"].unique()[0]],
                popup=self.create_HTML(c["Label"].unique()[0]),
                color=rgb2hex(cmap(norm(c["Label"].unique()[0]))),
                fill=True,
            ).add_to(m)

        m.save("Clusters.html")


if __name__=='__main__':
    c = clean_paloalto()
    data = c.clean_data()
    g = gridmap()
    #g.grid(data,7)
    p = plot()
    p.foliumplot() 