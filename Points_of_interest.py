import http.client
from Data_cleaning import clean_paloalto
import json
import pandas as pd
class POI:
    def __init__(self):
        self.conn = http.client.HTTPSConnection("api.foursquare.com")
        self.payload = ''
        self.headers = {}
    def getvenue(self,lat,long,radius,n):
        url = f"/v2/venues/search?client_id=VEZEHWT3U5FKX4KM0P2GVR33SW4ST2VQKC0RN5SYOJT3LGJ1%0A%0A&client_secret=V5GLVQJSK5HJHKGEYE0XQ3KY2CISS4SYZGGAPOVN5HOD0ZNU&v=20190425&ll={lat},{long}&intent=browse&radius={radius}&limit={n}"
        self.conn.request("GET",url , self.payload, self.headers)
        self.res = self.conn.getresponse()
        self.data = self.res.read()
        venue = json.loads(self.data.decode("utf-8"))["response"]["venues"]
        return venue
    
    def getpd(self,df):
        lat1,lon1 = self.getloc(df)
        name = []
        lat = []
        lon = []
        cat = []
        for j in range(len(lat1)):
            venue = self.getvenue(lat1[j],lon1[j],500,30)
            for i in range(len(venue)):
                try:
                    cat.append(venue[i]["categories"][0]["name"])
                    name.append(venue[i]["name"])
                    lat.append(venue[i]["location"]["lat"])
                    lon.append(venue[i]["location"]["lng"])
                except:
                    continue

        poipd = pd.DataFrame({"Name":name,"Latitude":lat,"Longitude":lon,"Category":cat})
        return poipd
    
    def getloc(self,df):
        lat = []
        lon = []
        pairloc = list(df["Pairlocation"].unique())
        pairloc = [i.split("x") for i in pairloc]
        
        for pair in pairloc:
            lat.append(float(pair[0]))
            lon.append(float(pair[1]))
        
        return lat,lon



if __name__=='__main__':
    c = clean_paloalto()
    data = c.clean_data()
    p = POI()
    #print(p.getloc(data))
    points_of_int = p.getpd(data)
    print(points_of_int.head)
    points_of_int.to_csv("points_of_int.csv")