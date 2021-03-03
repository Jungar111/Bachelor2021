import sys
sys.path.append(".")
from DataPrep.Data_cleaning import clean_paloalto
from sklearn.cluster import DBSCAN
import pandas as pd


class gridmap:
    def __init__(self):
        self.data = clean_paloalto().clean_data()
    
    # def addCenter(self, s):
    #     cluster = s["Label"]
    #     s["CenterLon"] = self.clusters[cluster][1]
    #     s["CenterLat"] = self.clusters[cluster][0]

    #     return s

    def getloc(self):
        lat = []
        lon = []
        pairloc = list(self.data["Pairlocation"].unique())
        pairloc = [i.split("x") for i in pairloc]
        
        for pair in pairloc:
            lat.append(float(pair[0]))
            lon.append(float(pair[1]))
        loc= []
        for i in range(len(lat)):
            loc.append([lat[i],lon[i]])
        return loc,lat,lon


    def grid(self):
        db = DBSCAN(eps=0.002,min_samples=1).fit(gridmap().getloc()[0])
        #kmeans = KMeans(n_clusters=c, random_state=0).fit(self.data[["Latitude","Longitude"]])
        #self.clusters=kmeans.cluster_centers_
        label = pd.DataFrame([gridmap().getloc()[1],gridmap().getloc()[2],db.labels_])
        #label.columns(["Latitude","Longitude","Label"])
        print(label)

        #griddf = self.data
        #merge(griddf,label)
        #griddf["Label"]=label
        #griddf = griddf.apply(self.addCenter, axis=1)
        #griddf.head()
        
        
    #     #return griddf#, self.clusters


if __name__=='__main__':
    g = gridmap()
    #print(g.getloc())
    #df, c = g.grid()
    print(g.grid())
    #df.to_csv("data/createdDat/CenteredData.csv")