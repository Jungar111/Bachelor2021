import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LinReg= np.array([26.29, 30.43, 33.33, 54.31, 54.57, 58.7, 58.61, 48.86])
Arima = np.array([23.19, 32.28, 31.68, 38.2, 45.33, 36.07, 40.39, 35.49])
LSTM =  np.array([28.56, 45.61, 41.52, 38.7, 33.63, 31.46, 40.10, 42.07])



VAR = np.array([18.36, 36.85, 35.09, 49.72, 61.65, 46.17, 52.05, 46.54])

#LSTM = np.array([28.44, 34.46, 43.33, 62.46, 55.13, 68.01, 78.44, 57.56])


plt.plot("Linear Regression", LinReg.mean(),"o", markersize=10, color = (72/255,83/255,88/255))
#plt.axvline("Linear Regression",0, ymax = 0.55, linestyle = "dotted", color = (72/255,83/255,88/255))

plt.plot("ARIMA", Arima.mean(), "o",markersize=10, color = (194/255,153/255,99/255))
#plt.axvline("ARIMA",0, ymax = 0.02, linestyle = "dotted", color = (194/255,153/255,99/255))

plt.plot("VAR", VAR.mean(), 'o', markersize=10,color = (152/255,159/255,166/255))
#plt.axvline("VAR",0, ymax = 0.45, linestyle = "dotted", color = (152/255,159/255,166/255))

plt.plot(r"$LSTM^*$", LSTM.mean(),"o", markersize=10, color = (152/255,180/255,180/255))
#plt.axvline(r"$LSTM^*$",0, ymax = 0.95, linestyle = "dotted", color = (152/255,180/255,180/255))

plt.title(r'Mean RMSE across clusters')
plt.ylabel("RMSE")
plt.show()


