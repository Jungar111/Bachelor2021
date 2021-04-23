import sys
sys.path.append(".")

from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from DataPrep.load_data import load_data
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


df = load_data()
df = df[df["Label"]==0]

pred = []
days_pred = []
train = int((len(df["Energy (kWh)"].index))*0.1)
re = len(df["Energy (kWh)"].index)-train
a = df.index[0]#.to_timestamp()
b = df.index[train]#.to_timestamp()
with tqdm(total=re, file=sys.stdout) as pbar:
    for i in range(re):
        try:
            sam = SARIMAX(np.sqrt(df["Energy (kWh)"][:b]),order=(6,1,7), freq="D")
            sam_fit = sam.fit(disp=-1)
            y_pred = sam_fit.forecast()
            pred.append(float(y_pred))
            days=1
            n = b + pd.Timedelta(days=days)
            days_pred.append(n)
        
        except np.linalg.LinAlgError:
            continue

        days = 1
        
        
        a = a + pd.Timedelta(days=days)
        b = b + pd.Timedelta(days=days)

        pbar.update(1)



pred = pd.DataFrame(pred)
pred.index = days_pred
pred.to_csv("data\createdDat\Arimares.csv")

plt.plot(pred)
plt.plot(df["Energy (kWh)"][pred.index[0]:pred.index[-1]])
plt.show()
print(f'r^2 score {r2_score(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],pred**2)}')
print(f'RMSE {np.sqrt(mean_squared_error(df["Energy (kWh)"][days_pred[0]:days_pred[-1]],pred**2))}')
