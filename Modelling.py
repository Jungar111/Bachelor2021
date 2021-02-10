import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import plotly.express as px
from Data_cleaning import clean_paloalto
class modelling:
    def lmmodels(self,df):
        from sklearn.linear_model import LinearRegression
        x = df["Charge Duration (mins)"].values.reshape(-1, 1)
        y =  df["Energy (kWh)"].values.reshape(-1, 1)
        lm1 = LinearRegression()
        lm1.fit(x,y) 
        Y_pred = lm1.predict(x)

        plt.scatter(x,y)
        plt.plot(x,Y_pred)
        plt.show()
    


if __name__=='__main__':
    c = clean_paloalto()
    m = modelling()
    data = c.clean_data()
    
    m.lmmodels(data)