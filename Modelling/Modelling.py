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
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge


class modelling:
    def  __init__(self):
        #self.df = importer().Import()
        self.df = importer().LagCreation()
        self.X_train,self.X_test, self.X_val,self.y_train,self.y_test, self.y_val = self.ttsplit(self.df)

    def lmmodels1(self):
        lm1 = LinearRegression()
        lm1.fit(self.X_train,self.y_train) 
        print(lm1.score(self.X_test,self.y_test))
    
    def ttsplit(self,df,target="Energy (kWh)"):
        cols = df.drop(columns=[target,"Start Date", 'Charging Time (mins)', 'Total Duration (mins)','ClusterID']).columns.to_list()
        X = df[cols]
        y = df[target]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.10, random_state=42)

        return X_train,X_test, X_val,y_train,y_test, y_val
    
    def neuralnet(self):

        model = Sequential()
        model.add(Dense(1000, input_dim=12, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='linear'))

        # compile the keras model
        model.compile(loss='mse', optimizer='adam')

        # fit the keras model on the dataset
        model.fit(self.X_train,self.y_train, epochs = 25, batch_size=256, validation_data=(self.X_val,self.y_val))

        # evaluate the keras model
        y_pred = model.predict(self.X_test)

        # evaluate predictions
        print("\nMAE=%f" % mean_absolute_error(self.y_test, y_pred))
        print("r^2=%f" % r2_score(self.y_test, y_pred))

        model.save("Models/NNWithLags.keras")
    
    def KernelRidge(self):
        kr = KernelRidge()
        kr.fit(self.X_train,self.y_train) 
        print(kr.score(self.X_test,self.y_test))

    def PCA(self):
        pca = PCA(10)
        self.X_train = pca.fit_transform(self.X_train)
        explvar = np.cumsum(pca.explained_variance_ratio_)
        print(pca.components_[0])
        print(len(pca.components_[0]))
        print(["Charging Time (mins)", "Total Duration (mins)", "Longitude", "Latitude", "Port Number", "Fee", "ClusterID"])
        plt.plot(explvar)
        plt.show()

        


if __name__=='__main__':
    m = modelling()
    m.KernelRidge()