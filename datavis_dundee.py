import matplotlib.pyplot as plt
import seaborn as sns
from dataclean_dundee import clean_dundee

"""
A class for visualizing the dundee data
"""


class datavis_dundee:
    def __init__(self):
        c = clean_dundee()
        self.dundee = c.clean_data()
    
    def pairs_plot(self):
        sns.pairplot(self.dundee)
        plt.show()


if __name__ == "__main__":
    d = datavis_dundee()
    dundee = d.dundee
    #d.pairs_plot()
    axes = plt.gca()
    axes.set_xlim([0,30])
    
    sns.histplot(dundee["Total kWh"], ax=axes)
    plt.show()
    #print(dundee.head())
    # plt.scatter(dundee["chargeTime"], dundee["Total kWh"])
    # plt.show()