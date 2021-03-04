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
        loc,lat,lon = self.getloc()
        db = DBSCAN(eps=0.002,min_samples=1).fit(loc)

        label = pd.DataFrame([lat,lon,db.labels_]).T
        label.columns=["Latitude","Longitude","Label"]
        

        griddf = self.data
        griddf = griddf.merge(label, on=["Latitude","Longitude"])
 
        
        
        return griddf


if __name__=='__main__':
    g = gridmap()
    #print(g.getloc())
    df = g.grid()
    
    df.to_csv("data/createdDat/CenteredData.csv")