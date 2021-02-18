import pandas as pd 
from MapBoxApi import MapBoxAPI
import matplotlib.pyplot as plt



class analysis:
    def __init__(self):
        self.data = pd.read_csv("points_of_int1.csv")
    
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
        self.data = pd.read_csv("points_of_int1.csv")
        self.data=self.data[(self.data["Longitude"]<-122.10) & (self.data["Longitude"]>-122.2)]
    
    # def categories(self,data):
    #     data=data.reset_index()
    #     for i in range(len(data["Category"])):
    #         if re.search("^.*Food.*$", data["Category"][i]):
    #             print(re.search("^.*Food.*$", data["Category"][i]))
    #             print(i)
                
    #             #data["Category"][i]="Resturant"

    

if __name__=='__main__':
    a = analysis()
    c = clean()
    data=c.data
    #c.categories(data)
    a.sjov(data)
    #print((data["Category"].unique()))
    
    










