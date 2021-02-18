from Data_cleaning import clean_paloalto
import pandas as pd
import platform
import time
import matplotlib.pyplot as plt

class LagCreation:
    def __init__(self):
        self.data = clean_paloalto().clean_data()
        self.data = self.data[(self.data["End Date"].dt.year < 2017) & (self.data["Start Date"].dt.year > 2015)]
        self.data = self.data[self.data['Station Name'] == 'PALO ALTO CA / WEBSTER #2']

    def bucket(self):
        data = self.data.resample('2H', on='Start Date').agg(({'Energy (kWh)':'sum','Charge Duration (mins)':'sum', 'Fee':'sum'}))
        return data
    
    

if __name__ == "__main__":
    l = LagCreation()
    start = time.time()
    data = l.bucket()
    print(data)
    plt.bar(data.groupby(data.index.hour).sum().index,data.groupby(data.index.hour).sum()["Energy (kWh)"])
    plt.show()

    end = time.time()
    print(end - start)