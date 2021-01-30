import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class clean_dundee:
    def __init__(self):
        self.datapaths = ["data/Dundee/cp-data-dec-mar-2018.csv", "data/Dundee/cp-data-mar-may-2018.csv", "data/Dundee/cpdata.csv"]

    def merge_data(self):
        dec_mar_dat = pd.read_csv(self.datapaths[0])
        mar_may_dat = pd.read_csv(self.datapaths[1])
        cp_dat = pd.read_csv(self.datapaths[2])

        dundee = cp_dat.append(dec_mar_dat.append(mar_may_dat))
        return dundee
    
    def clean_data(self):
        dundee = self.merge_data()

        dundee = dundee.dropna()

        return dundee
    
    def pairs_plot(self):
        dundee = self.clean_data()
        sns.pairplot(dundee)
        plt.show()



if __name__ == '__main__':
    c = clean_dundee()
    dundee = c.clean_data()

    print(dundee.describe())

    