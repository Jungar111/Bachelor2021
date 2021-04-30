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


def ArimaModels(df):
    pred = []
    days_pred = []
    train = int((len(df["Energy (kWh)"].index))*0.1)
    re = len(df["Energy (kWh)"].index)-train
    a = df.index[0]#.to_timestamp()
    b = df.index[train]#.to_timestamp()
    with tqdm(total=re, file=sys.stdout) as pbar:
        for i in range(re):
            try:
                sam = SARIMAX(df["Energy (kWh)"][:b], order=(9,1,7) ,freq="D")
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
            pbar.update(1)
    pred = pd.DataFrame(pred)
    pred.index = days_pred

    
    plt.plot(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],label="Actual", alpha=0.6)
    plt.plot(pred[:-1], color="red", label="Arima Prediction", alpha=0.6)
    plt.legend()
    print(f'r^2 score {r2_score(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],pred[:-1])}')
    print(f'RMSE {np.sqrt(mean_squared_error(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],pred[:-1]))}')
    plt.show()
    return pred, days_pred

df = load_data()
df = df[df["Label"]==1]
df = df[df.index[df["Energy (kWh)"]>0][0]:]
df = df.drop(columns=["Charging Time (mins)","Parking Time (mins)"])
print("Imported!")

pred, days_pred =  ArimaModels(df)
pred.to_csv("data\createdDat\ARIMA_Prediction_Label1.csv")


print("DONE!")