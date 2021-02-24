import http.client


class MapBoxAPI:
    def __init__(self):
        self.base_url = "https://api.mapbox.com/styles/v1/mapbox/light-v10/static/"
        self.access_token = "pk.eyJ1IjoiYXNnZXJ0YW5nIiwiYSI6ImNra3M0Z25uYjEzNmkydnMxcW91ZTA3djYifQ.4t-EqKfxIzyAo1uHW6232w"
    
    def get_image(self, long, lat, name, size):
        w,h = self.image_dim(lat, long, size)
        
        url = self.base_url + f"[{min(long)},{min(lat)},{max(long)},{max(lat)}]/{w}x{h}?access_token={self.access_token}"
        
        #url = "https://api.mapbox.com/styles/v1/mapbox/streets-v11/static/[-122.164239,37.421104,-122.11148,37.450375]/901x500?access_token=pk.eyJ1IjoiYXNnZXJ0YW5nIiwiYSI6ImNra3M0Z25uYjEzNmkydnMxcW91ZTA3djYifQ.4t-EqKfxIzyAo1uHW6232w"
        conn = http.client.HTTPSConnection("api.mapbox.com")
        payload = ''
        headers = {}
        conn.request("GET", url, payload, headers)
        res = conn.getresponse()
        data = res.read()
        if res.getcode() == 200:
            file = open(f'img/{name}', "wb")
            file.write(data)
            file.close()
        else:
            print(res.getcode())

    def image_dim(self, lat, lon, size):
        latlen = lat[1] - lat[0]
        lonlen = lon[1] - lon[0]
        

        if lonlen > latlen:
            ref = lonlen/latlen
            return (int(size*ref), size)
            
        elif lonlen < latlen:
            ref = latlen/lonlen
            return (size, int(size*ref))


if __name__ == "__main__":
    api = MapBoxAPI()
    lat = (-77.043686,-77.028923)
    lon = (38.892035,38.904192)
    api.get_image(lat, lon, "test.png",500)
    #api.test2()