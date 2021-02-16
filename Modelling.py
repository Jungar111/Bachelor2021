import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import plotly.express as px
import matplotlib.patches as mpatches

from Data_cleaning import clean_paloalto
class modelling:
    def lmmodels(self,df):
        from sklearn.linear_model import LinearRegression
        x = df["Charge Duration (mins)"].values.reshape(-1, 1)
        y =  df["Energy (kWh)"].values.reshape(-1, 1)
        #lm1 = LinearRegression()
        #lm1.fit(x,y) 
        #Y_pred = lm1.predict(x)
        

class plot:
    def poiplot(self,df):    
        purple_patch = mpatches.Patch(color="Purple", label='Plug type = J1772')
        yellow_patch = mpatches.Patch(color="Yellow", label='Plug type = NEMA 5-20R')

        plt.scatter(df["Charge Duration (mins)"],df["Energy (kWh)"], c=pd.factorize(df["Plug Type"])[0])
        plt.xlabel("Charge duration (min)")
        plt.ylabel("Energy (kWh)")
        plt.title("Energy vs charge duration")
        plt.legend(handles=[purple_patch,yellow_patch])
        plt.show()
    


if __name__=='__main__':
    c = clean_paloalto()
    m = modelling()
    p = plot()
    data = c.clean_data()
    #m.lmmodels(data)
    p.poiplot(data)
