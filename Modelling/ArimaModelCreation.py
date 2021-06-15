import sys
sys.path.append(".")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from DataPrep.load_data import load_data
from DataPrep.LagCreation import lags
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")


def ArimaModels(df,ar,d,ma):
    pred = []
    days_pred = []
    param = pd.DataFrame(columns=["ar.L1","ar.L2","ar.L3","ar.L4","ar.L5","ar.L6","ar.L7","ar.L8","ar.L9",
    "ma.L1","ma.L2","ma.L3","ma.L4","ma.L5","ma.L6","ma.L7","ma.L8","ma.L9","sigma2","Label"])
    train = int((len(df["Energy (kWh)"].index))*0.8)
    re = len(df["Energy (kWh)"].index)-train
    a = df.index[0]#.to_timestamp()
    b = df.index[train]#.to_timestamp()
    with tqdm(total=re, file=sys.stdout) as pbar:
        for i in range(re):
            try:
                sam = SARIMAX(df["Energy (kWh)"][:b], order=(ar,d,ma), seasonal_order=(1,1,1,14) ,freq="D")
                sam_fit = sam.fit(disp = False, full_output = False)

                days = 1
                n = b + pd.Timedelta(days=days)
                y_pred = sam_fit.forecast()
                #n1 = n + pd.Timedelta(days=1)
                #y_pred = sam_fit.forecast(steps = days)
                #y_pred = sam_fit.predict(start=n,end=n,typ="levels")
                #y_pred.index=y_pred.index.to_timestamp()
                
                pred.append(float(y_pred))

                a = a + pd.Timedelta(days=days)
                b = b + pd.Timedelta(days=days)
                days_pred.append(n)
            except np.linalg.LinAlgError:
                days = 1
                n = b + pd.Timedelta(days=days)
                pred.append(y_pred[-1])

                a = a + pd.Timedelta(days=days)
                b = b + pd.Timedelta(days=days)
                days_pred.append(n)
            p = {} 
            for j in range(1,9):
                try:
                    p[f"ar.L{j}"] = sam_fit.params[j-1]
                except:
                    p[f"ar.L{j}"] = np.nan
                try:
                    p[f"ma.L{j}"] = sam_fit.params[j+ar]
                except:
                    p[f"ma.L{i}"] = np.nan
            p["sigma2"]= sam_fit.params[-1]
            p["Label"] = df["Label"][0]

            param = param.append(p, ignore_index=True)
            pbar.update(1)
    pred = pd.DataFrame(pred)
    pred.index = days_pred

    
    # plt.plot(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],label="Actual", alpha=0.6)
    # plt.plot(pred, color="red", label="Arima Prediction", alpha=0.6)
    # plt.legend()
    # print(f'r^2 score {r2_score(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],pred)}')
    # print(f'RMSE {np.sqrt(mean_squared_error(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],pred))}')
    # plt.show()
    return pred, days_pred, param


ar = [7,9,8,7,8,8,7,9]
d = [1,1,1,1,1,1,1,1]
ma = [8,7,9,8,9,9,8,9]
for i in range(1):
    df = load_data()
    df = df[df["Label"]==i]
    df = df[df.index[df["Energy (kWh)"]>0][0]:]
    df = df.drop(columns=["Charging Time (mins)","Parking Time (mins)"])
    print("Imported!")

    pred, days_pred, param =  ArimaModels(df,ar[i],d[i],ma[i])
    pred.to_csv(f"data\createdDat\ARIMA_Prediction_s_Label{i}.csv")
    param.to_csv(f"data\createdDat\ARIMA_Params_s_Label{i}.csv")    

print("DONE!")