import sys
sys.path.append(".")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import plotly.express as px
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from DataPrep.ImportData import importer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf



class modelling:
    def lmmodels1(self,df):
        X_train,X_test,X_val,y_train,y_test,y_val = self.ttsplit(df)
        lm1 = LinearRegression()
        lm1.fit(X_train,y_train) 
        print(lm1.score(X_test,y_test))
    
    def ttsplit(self,df,target="Energy (kWh)"):
        cols = df.drop(columns=[target,"Start Date"]).columns.to_list()
        X = df[cols]
        y = df[target]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.10)

        return X_train,X_test, X_val,y_train,y_test, y_val
    
    def neuralnet(self,df):
        X_train,X_test, X_val,y_train,y_test, y_val = self.ttsplit(df)

        model = Sequential()
        model.add(Dense(1000, input_dim=12, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='relu'))

        # compile the keras model
        model.compile(loss='mse', optimizer='adam')

        # fit the keras model on the dataset
        model.fit(X_train,y_train, epochs = 25, batch_size=128, validation_data=(X_val,y_val))

        # evaluate the keras model
        y_pred = model.predict(X_test)

        # evaluate predictions
        print("\nMAE=%f" % mean_absolute_error(y_test, y_pred))
        print("r^2=%f" % r2_score(y_test, y_pred))

        model.save("Models/NNWithLags.keras")
        
        
        

class plot:
    def poiplot(self,df):    
        purple_patch = mpatches.Patch(color="Purple", label='Plug type = J1772')
        yellow_patch = mpatches.Patch(color="Yellow", label='Plug type = NEMA 5-20R')

        #plt.scatter(x,y,c=df["Port Type"])

        plt.hexbin(x,y, gridsize=50)
        #plt.plot(x,Y_pred, c="red")
        plt.show()
    

        # y_pred = lm1.predict(x)

        # plt.scatter(x,y,c=pd.factorize(df["Port Type"])[0], alpha=0.3)
        # #plt.scatter(x,y)

        # #plt.hexbin(x,y, gridsize=50)
        # #plt.plot(x,Y_pred, c="red")
        # plt.show()
    


if __name__=='__main__':
    m = modelling()
    p = plot()
    data = importer().LagCreation()
    
    m.neuralnet(data)