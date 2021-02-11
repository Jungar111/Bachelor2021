import http.client
from Data_cleaning import clean_paloalto
class POI:
    conn = http.client.HTTPSConnection("api.foursquare.com")
    payload = ''
    headers = {}
    conn.request("GET", "/v2/venues/search?client_id=VEZEHWT3U5FKX4KM0P2GVR33SW4ST2VQKC0RN5SYOJT3LGJ1%0A%0A&client_secret=V5GLVQJSK5HJHKGEYE0XQ3KY2CISS4SYZGGAPOVN5HOD0ZNU&v=20190425&ll=40.7099,-73.9622&intent=browse&radius=500&limit=10", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))

if __name__=='__main__':
    c = clean_paloalto()
    data = c.clean_data()
    p = POI()
