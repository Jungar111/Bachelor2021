import sys
sys.path.append(".")
from sklearn.linear_model import LinearRegression
from DataPrep.ImportData import importer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class lm:
    def  __init__(self):
        #self.df = importer().Import()
        self.df = importer().LagCreation()
        
        self.X_train,self.X_test, self.X_val,self.y_train,self.y_test, self.y_val = self.ttsplit(self.df)
        print(self.X_train.columns)




    def lmmodels1(self):
        lm1 = LinearRegression()
        lm1.fit(self.X_train,self.y_train) 
        y_pred=lm1.predict(self.X_train)
        print("\nRMSE=%f" % np.sqrt(mean_squared_error(self.y_test, y_pred)))
        print("r^2=%f" % r2_score(self.y_test, y_pred))
        print(dict(zip(self.X_train.columns,lm1.coef_)))
        plt.plot(self.y_train,y_pred)
        plt.show()
        
        #plt.scatter(self.y_test,y_pred)
        #plt.show()
        #plt.plot(y_pred)
        #plt.show()
        
    
    def ttsplit(self,df,target="Energy (kWh)"):
        cols = df.drop(columns=[target,"Start Date", 'Charging Time (mins)', 'Total Duration (mins)',"Port Number","Level 1","Level 2","Energy (kWh)_lag1",'Energy (kWh)_lag2','Energy (kWh)_lag3','Energy (kWh)_lag4','Energy (kWh)_lag5',"Label"]).columns.to_list()
        X = df[["Energy (kWh)_lag1",'Energy (kWh)_lag2','Energy (kWh)_lag3','Energy (kWh)_lag4','Energy (kWh)_lag5']]
        #print(X.head())
        y = df[target]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.10, random_state=42)

        return X_train,X_test, X_val,y_train,y_test, y_val

if __name__=='__main__':
    m = lm()
    #m.lmmodels1()
    print(m.lmmodels1())

# Lags+date+clusterid = r^2 = 0.41, coef are shit
# date+clusterid+portnumber+level1/2 = r^2 = 0.415, coef are fucked 
# date + clusterid = r^2 = 0.37, clusterid coef very low and date very high 
# only lags = R^2 = 0.35, but fine coef
# lags+ level  = R^2 = 0.137, but fine coef
# lags+ level+ port number  = R^2 = 0.37, but fine coef
# lags+ level+ port number+ year month  = R^2 = 0.37, but fine coef
# lags+ level+ port number+ year month + Cluster id = R^2 = 0.436, but fine coef

