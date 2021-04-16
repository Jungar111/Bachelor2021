import sys
sys.path.append(".")

from DataPrep.ImportData import importer
from Modelling import modelling
from sklearn.metrics import mean_squared_error, r2_score
from DataPrep.LagCreation import lags
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


df = importer().Import()

df2 = pd.DataFrame()

df = df.set_index("Start Date")
df.index = df.index.to_period("D")

idx = pd.period_range(df.index.min(),df.index.max())
for i in range(8): 
    dat = df[df["Label"]==float(i)].reindex(idx, fill_value=0)
    dat.Label = float(i)
    df2 = df2.append(dat)

df2 = df2.sort_index()
df2["Start Date"] = df2.index

y = df2[["Energy (kWh)","Start Date", "Label"]]

y_cols = pd.DataFrame(columns=["Label 0","Label 1","Label 2","Label 3","Label 4","Label 5","Label 6","Label 7"])
for i in range(8):
    y_cols[f"Label {i}"]=y["Energy (kWh)"][y["Label"]==float(i)]

pred = []
days_pred = []
train = int((len(y_cols["Label 0"].index))*0.01)
re = len(y_cols["Label 0"].index)-train
a = y_cols.index[0].to_timestamp()
b = y_cols.index[train].to_timestamp()
for i in range(10):#re):
    sam = SARIMAX(y_cols["Label 0"][:b],order=(7,1,5), freq="D")
    sam_fit = sam.fit(disp=0)

    days = 1
    n=b+pd.Timedelta(days=days)
    n1 = n + pd.Timedelta(days=1)
    y_pred = sam_fit.forecast(steps = days)
    y_pred.index=y_pred.index.to_timestamp()
    
    pred.append(float(y_pred))

    b = b + pd.Timedelta(days=days)
    days_pred.append(n)
    print(f"Progress {i}:{re}")



pred = pd.DataFrame(pred)
pred.index = days_pred
pred.to_csv("Arimares.csv")

plt.plot(pred)
plt.plot(y_cols["Label 0"][pred.index[0]:pred.index[-1]])
plt.show()
print(r2_score(y_cols["Label 0"][pred.index[0]:pred.index[-1]],pred))
