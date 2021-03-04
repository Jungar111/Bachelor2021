import sys
sys.path.append(".")
import folium
from DataPrep.grid import gridmap
import matplotlib.cm as cm
from matplotlib.colors import Normalize, rgb2hex

class plotClusters:
    def create_HTML(self,cluster):
        html=f"<h4> Cluster: {cluster}</h4>"
        return html

    def foliumplot(self):
        m = folium.Map(location=[37.435, -122.16], tiles="Stamen Toner", zoom_start=13)
        g = gridmap()
        grid = g.grid().groupby("Pairlocation")

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

        m.save("Vizualisation\FoliumPlots\Clusters.html")

if __name__=='__main__':
    p = plotClusters()
    p.foliumplot()