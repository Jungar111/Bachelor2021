import pandas as pd 
from MapBoxApi import MapBoxAPI
import matplotlib.pyplot as plt

class analysis:
    def __init__(self):
        self.data = pd.read_csv("points_of_int.csv")
    
    def sjov(self,data):
        fig, ax = plt.subplots()
        BBox = (data.Longitude.min()- 0.001, data.Longitude.max() + 0.001, data.Latitude.min() - 0.001, data.Latitude.max() + 0.001)

        name = "PaloAlto.png"
        api = MapBoxAPI()
        api.get_image((BBox[0], BBox[1]), (BBox[2], BBox[3]), name, 500)

        paloaltoimg = plt.imread(f'img/{name}')

        plt.scatter(data["Longitude"],data["Latitude"], alpha=0.6,s=5)
        plt.title("Points of interest")
        plt.imshow(paloaltoimg, zorder=0, extent = BBox, aspect= 'equal')
        plt.show()

class clean:
    def __init__(self):
        self.data = pd.read_csv("points_of_int.csv")
        self.data=self.data[(self.data["Longitude"]<-122.10) & (self.data["Longitude"]>-122.2)]
    

if __name__=='__main__':
    a = analysis()
    c = clean()
    data=c.data
    #a.sjov(data)
    print(data["Category"].unique())

    










