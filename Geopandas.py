import geopandas
from Data_cleaning import clean_paloalto
class geo:
    def map(self):
        df = clean_paloalto().clean_data()
        gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude), 
                                    crs="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
        print(gdf.head())


if __name__=='__main__':
    c = clean_paloalto()
    data = c.clean_data()
    g = geo()
    print(g.map())