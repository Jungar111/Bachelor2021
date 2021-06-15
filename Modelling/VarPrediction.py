import sys
sys.path.append(".")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
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
    pred = pd.DataFrame(columns=["Label0","Label1","Label2","Label3","Label4","Label5","Label6","Label7"])
    days_pred = []
    param = pd.DataFrame(columns=["Label","ar.L1","ar.L2","ar.L3","ar.L4","ar.L5","ar.L6","ar.L7","ar.L8","ar.L9",
    "ma.L1","ma.L2","ma.L3","ma.L4","ma.L5","ma.L6","ma.L7","ma.L8","ma.L9","sigma2"])
    train = int((len(df.index))*0.1)
    re = len(df.index)-train
    a = df.index[train-150]#.to_timestamp()
    b = df.index[train]#.to_timestamp()
    with tqdm(total=re, file=sys.stdout) as pbar:
        for i in range(re):
            try:
                sam = VAR(df[:b])
                sam_fit = sam.fit(maxlags=8)
                


                days = 1
                n = b + pd.Timedelta(days=days)
                y_pred = sam_fit.forecast(sam_fit.y,steps=1)
                #y_pred = pd.DataFrame(y_pred, columns=["Label0","Label1","Label2","Label3","Label4","Label5","Label6","Label7"])
                pred.loc[i]=y_pred[0]


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
            
            #p["sigma2"]= sam_fit.params[-1]


            #param = param.append(p, ignore_index=True)
            pbar.update(1)

    
    # plt.plot(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],label="Actual", alpha=0.6)
    # plt.plot(pred, color="red", label="Arima Prediction", alpha=0.6)
    # plt.legend()
    # print(f'r^2 score {r2_score(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],pred)}')
    # print(f'RMSE {np.sqrt(mean_squared_error(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],pred))}')
    # plt.show()
    return pred, days_pred, param




df = load_data()
df1 = pd.DataFrame(columns=["Label0","Label1","Label2","Label3","Label4","Label5","Label6","Label7"])
for j in range(8):
    df1[f"Label{j}"] = df["Energy (kWh)"][df["Label"]==j]
    


print("Imported!")

pred, days_pred, param =  ArimaModels(df1.diff().dropna())
pred["days_pred"] = days_pred
pred = pred.set_index("days_pred") 

pred.to_csv("VarPred_correct.csv")




# for i in range(8):
#     pred_i = pred[f"Label{i}"][pred.index[pred[f"Label{i}"]>0][0]:]
#     plt.plot(df1[f"Label{i}"][-len(pred_i)+1:], label="Actual", alpha=0.6)
#     plt.plot(df1.index[-len(pred_i)+1:],pred_i[:-1], label= "Predicted",alpha=0.6)
#     plt.title(f"Multivatiate prediction label{i}")
#     plt.legend()
#     plt.show()

print("DONE!")