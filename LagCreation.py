import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import plotly.express as px
from Data_cleaning import clean_paloalto

# class modelling:
#     def lmmodels(self,df):
#         from sklearn.linear_model import LinearRegression
#         x = df["Charge Duration (mins)"].values.reshape(-1, 1)
#         y =  df["Energy (kWh)"].values.reshape(-1, 1)
#         lm1 = LinearRegression()
#         lm1.fit(x,y) 
#         Y_pred = lm1.predict(x)

#         #plt.scatter(x,y,c=df["Port Type"])

#         plt.hexbin(x,y, gridsize=50)
#         #plt.plot(x,Y_pred, c="red")
#         plt.show()
    

#     def lmmodels1(self,df):
#         from sklearn.linear_model import LinearRegression
#         x = df["Charge Duration (mins)"].values.reshape(-1, 1)
#         y =  df["Energy (kWh)"].values.reshape(-1, 1)
#         lm1 = LinearRegression()
#         lm1.fit(x,y) 
#         Y_pred = lm1.predict(x)

#         plt.scatter(x,y,c=pd.factorize(df["Port Type"])[0], alpha=0.3)
#         #plt.scatter(x,y)

#         #plt.hexbin(x,y, gridsize=50)
#         #plt.plot(x,Y_pred, c="red")
#         plt.show()
    


if __name__=='__main__':
    c = clean_paloalto()
#    m = modelling()
    data = c.clean_data()
    print(data.head())
    #m.lmmodels(data)
#    m.lmmodels1(data)