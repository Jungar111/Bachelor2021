import matplotlib.pyplot as plt
import seaborn as sns
from dataclean_dundee import clean_dundee

"""
A class for visualizing the dundee data
"""


class datavis:
    def __init__(self):
        pass
    
    def pairs_plot(self, df):
        sns.pairplot(df)
        plt.show()
    
    def distplot(self, df, shapex, shapey):
        fig, ax = plt.subplots(shapex,shapey)
        cols = list(df.columns)
        fig.suptitle('Distributions of variables')
        k = 0
        for i in range(shapex):
            for j in range(shapey):
                if "time" not in cols[k]:
                    ax[i,j].hist(df[cols[k]])
                    ax[i,j].title.set_text(cols[k])
                    k += 1
                else:
                    continue
        
        plt.show()

        


if __name__ == "__main__":
    d = datavis()
    c = clean_dundee()
    d.distplot(c.clean_data(), 3, 5)
    
    
    #d.pairs_plot()
    axes = plt.gca()
    axes.set_xlim([0,30])
    
    #sns.histplot(dundee["Total kWh"], ax=axes)
    #plt.show()
    #print(dundee.head())
    # plt.scatter(dundee["chargeTime"], dundee["Total kWh"])
    # plt.show()