from DataPrep.Data_cleaning import clean_paloalto
from sklearn.cluster import KMeans
import pandas as pd


class gridmap:
    def __init__(self):
        self.data = clean_paloalto().clean_data()
    
    def addCenter(self, s):
        cluster = s["Label"]
        s["CenterLon"] = self.clusters[cluster][1]
        s["CenterLat"] = self.clusters[cluster][0]

        return s

    def grid(self,c = 8):
        
        kmeans = KMeans(n_clusters=c, random_state=0).fit(self.data[["Latitude","Longitude"]])
        self.clusters=kmeans.cluster_centers_
        label = kmeans.labels_
        
        griddf = self.data
        griddf["Label"]=label
        griddf = griddf.apply(self.addCenter, axis=1)
        griddf.head()
        
        
        return griddf, self.clusters


if __name__=='__main__':
    g = gridmap()
    df, c = g.grid()
    df.to_csv("data/createdDat/CenteredData.csv")