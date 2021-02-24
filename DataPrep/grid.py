from DataPrep.Data_cleaning import clean_paloalto
import folium
from sklearn.cluster import KMeans

class gridmap:
    def grid(self,df, c):

        kmeans = KMeans(n_clusters=c, random_state=0).fit(df[["Latitude","Longitude"]])
        clusters=kmeans.cluster_centers_
        return clusters
        



if __name__=='__main__':
    c = clean_paloalto()
    data = c.clean_data()
    g = gridmap()
    # print(g.grid(data))