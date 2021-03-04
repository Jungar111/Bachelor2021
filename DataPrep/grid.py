import sys
sys.path.append(".")
from DataPrep.Data_cleaning import clean_paloalto
from sklearn.cluster import KMeans, DBSCAN
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
        
        dbscan = DBSCAN().fit(self.data[["Latitude","Longitude"]])
        
        label = dbscan.labels_
        
        griddf = self.data
        griddf["Label"]=label
        #griddf = griddf.apply(self.addCenter, axis=1)
        #griddf.head()
        
        
        return griddf #self.clusters


if __name__=='__main__':
    g = gridmap()
    df = g.grid()
    df.head()
    #df.to_csv("data/createdDat/CenteredData.csv")