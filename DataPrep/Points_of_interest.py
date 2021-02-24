import http.client
from Data_cleaning import clean_paloalto
import json
import pandas as pd
class POI:
    def __init__(self):
        self.conn = http.client.HTTPSConnection("api.foursquare.com")
        self.payload = ''
        self.headers = {}

    def getcategory(self):
        url1 = "/v2/venues/categories?client_id=VEZEHWT3U5FKX4KM0P2GVR33SW4ST2VQKC0RN5SYOJT3LGJ1%0A%0A&client_secret=V5GLVQJSK5HJHKGEYE0XQ3KY2CISS4SYZGGAPOVN5HOD0ZNU&v=20190425"
        self.conn.request("GET", url1, self.payload, self.headers)
        self.res = self.conn.getresponse()
        self.data = self.res.read()
        categories = json.loads(self.data.decode("utf-8"))["response"]["categories"]
        return categories

    
    def getvenue(self,lat,long,radius,n, id):
        url = f"/v2/venues/search?client_id=VEZEHWT3U5FKX4KM0P2GVR33SW4ST2VQKC0RN5SYOJT3LGJ1%0A%0A&client_secret=V5GLVQJSK5HJHKGEYE0XQ3KY2CISS4SYZGGAPOVN5HOD0ZNU&v=20190425&ll={lat},{long}&intent=browse&radius={radius}&limit={n}&categoryId={id}"
        self.conn.request("GET",url , self.payload, self.headers)
        self.res = self.conn.getresponse()
        self.data = self.res.read()
        venue = json.loads(self.data.decode("utf-8"))["response"]["venues"]
        return venue
    
    def getpd(self):
        # Den bruger da ikke data????
        #lat1,lon1 = self.getloc(df)
        lat1 = [37.450375-0.005*i for i in range(7)]
        lon1 = [-122.11148-0.005*i for i in range(12)]
        
        categories = self.getcategory()

        name = []
        lat = []
        lon = []
        cat = []
        subcat =[]
        #notimportant = ["Parking","Tree","Road"]
        for j in range(len(lat1)):
            for k in range(len(lon1)):
                for q in range(len(categories)):
                    venue = self.getvenue(lat1[j],lon1[k],500,50,categories[q]["id"])
                    for i in range(len(venue)):
                        try:
                            #if venue[i]["categories"][0]["name"] not in notimportant:
                            #cat.append(venue[i]["categories"][0]["name"])
                            subcat.append(venue[i]["categories"][0]["name"])
                            cat.append(categories[q]["name"])
                            name.append(venue[i]["name"])
                            lat.append(venue[i]["location"]["lat"])
                            lon.append(venue[i]["location"]["lng"])
                        except:
                            continue

        poipd = pd.DataFrame({"Name":name,"Latitude":lat,"Longitude":lon,"Category":cat,"Sub Category":subcat})
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
    #c = clean_paloalto()
    #data = c.clean_data()
    p = POI()
    #print(p.getloc(data))
    #print(p.getcategory())
    
    points_of_int1 = p.getpd()
    print(points_of_int1.head)
    points_of_int1.to_csv("data/createdDat/points_of_int.csv")