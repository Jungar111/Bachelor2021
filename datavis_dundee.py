import matplotlib.pyplot as plt
from scipy.stats.morestats import median_test
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
                try:
                    if "date" in cols[k].lower() or "Group" in cols[k] or "Site" in cols[k] or "Model" in cols[k]:
                        ax[i,j].hist(df[cols[k]])
                        ax[i,j].set_xticks([])
                    else:
                        ax[i,j].hist(df[cols[k]])
                    ax[i,j].title.set_text(cols[k])
                    k += 1
                except:
                    ax[i,j].title.set_text(cols[k])
                    k += 1
        
        plt.show()
    
    def corrPlot(self, df, *cols):
        sns.pairplot(df[list(cols)])
        plt.show()

        


if __name__ == "__main__":
    d = datavis()
    c = clean_dundee()
    
    #d.distplot(c.clean_data(), 3, 5)

    d.corrPlot(c.clean_data(),("datetime_start", "Total kWh"))
    
    
    #d.pairs_plot()
    
    #sns.histplot(dundee["Total kWh"], ax=axes)
    #plt.show()
    #print(dundee.head())
    # plt.scatter(dundee["chargeTime"], dundee["Total kWh"])
    # plt.show()