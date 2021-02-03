import requests

class googleMapsAPI:
    def __init__(self):
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap?"
        self.secret_key = "AIzaSyAmowxq248aC_ApgDJbLqskuOxbhpsgE98"
    
    def get_image(self, zoom, size, maptype, *args):
        url = self.base_url
        for arg in args:
            url = url + arg + ','
        
        url = url[:len(url) - 1]
        url = url + f'&zoom={zoom}&size={size[0]}x{size[1]}&maptype={maptype}&key={self.secret_key}'
        
        r = requests.get(url)
        print(r)

api = googleMapsAPI()
api.get_image(200,(200,200), "roadmap","Viveterpvej 2","9560","Hadsund")
