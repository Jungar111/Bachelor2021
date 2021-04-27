import sys
sys.path.append(".")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from DataPrep.load_data import load_data
from DataPrep.LagCreation import lags
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

def smape(A, F):
    return 100/len(A) * np.sum( np.abs(F - A) / (np.abs(A) + np.abs(F)))


def ArimaModels(df):
    res = pd.DataFrame(columns=["Label","ar","d","ma","AIC","r^2","RMSE","MAPE","MAE","SMAPE"])
    with tqdm(total=2100, file=sys.stdout) as pbar:
        for label in range(7):
            df1=df[df["Label"]==label]
            #df1=df1.sort_values("Start Date")
            #y = df1[["Energy (kWh)","Start Date"]].set_index("Start Date",drop=False)

            #y= y.resample("d").min()
            #X = lags().buildLaggedFeatures(y["Energy (kWh)"], ["Energy (kWh)"],5, dropna=False)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)
            #X_train=X_train.drop(columns="lag_0")
            #X_test=X_test.drop(columns="lag_0")
            #y_train = y_train.set_index("Start Date",drop=False)
            #y_test = y_test.set_index("Start Date",drop=False)

            #y_train.freq= "D"
            #y_train = y_train.resample("d").min()
            #y_test = y_test.resample("d").min()
            #y_train["Energy (kWh)"]  = y_train["Energy (kWh)"].fillna(0)
            #y_test["Energy (kWh)"]  = y_test["Energy (kWh)"].fillna(0)

            test = int(len(df1["Energy (kWh)"])*0.8)
            
            for ar in range(10):
                for d in range(3):
                    for ma in range(10):
                        try:
                            sam = SARIMAX((df1["Energy (kWh)"][:test]), order=(ar,d,ma), trend="n", freq="D")
                            sam_fit = sam.fit(method="lbfgs", disp = False, full_output = False)
                            n=len(df1[test+1:])
                            y_pred = sam_fit.forecast(steps = n)

                            AIC = sam_fit.aic
                            r = r2_score(df1["Energy (kWh)"][test+1:],y_pred)
                            RMSE = np.sqrt(mean_squared_error(df1["Energy (kWh)"][test+1:],y_pred))
                            MAPE = mean_absolute_percentage_error(df1["Energy (kWh)"][test+1:],y_pred)
                            MAE = mean_absolute_error(df1["Energy (kWh)"][test+1:],y_pred)
                            SMAPE = smape(df1["Energy (kWh)"][test+1:],y_pred)
                        
                            results = dict(zip(list(res.columns),[label,ar,d,ma,AIC,r,RMSE,MAPE,MAE,SMAPE]))
                        except np.linalg.LinAlgError:
                            results = dict(zip(list(res.columns),[label,ar,d,ma,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]))
                        res = res.append(results,True)
                        pbar.update(1)

        
    
    return res

df = load_data()

print("Imported!")

models = ArimaModels(df)

models.to_csv("ArimaModels.csv")

print("DONE!")