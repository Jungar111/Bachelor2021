import sys
sys.path.append(".")

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
from DataPrep.load_data import load_data
from DataPrep.LagCreation import lags
from Tobit import Tobit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def censor(s):
    energy = s["Energy (kWh)"] 
    # energy_lag1 = s["Energy (kWh)_lag1"]
    # energy_lag2 = s["Energy (kWh)_lag2"]
    # energy_lag3 = s["Energy (kWh)_lag3"]
    # energy_lag4 = s["Energy (kWh)_lag4"]
    # energy_lag5 = s["Energy (kWh)_lag5"]

    s["Censored"] = False
    if energy > 200:
        s["Energy (kWh)"] = 200
        s["Censored"] = True
    # if energy_lag1 > 0.5:
    #     s["Energy (kWh)_lag1"] = 0.5
    # if energy_lag2 > 0.5:
    #     s["Energy (kWh)_lag2"] = 0.5
    # if energy_lag3 > 0.5:
    #     s["Energy (kWh)_lag3"] = 0.5
    # if energy_lag4 > 0.5:
    #     s["Energy (kWh)_lag4"] = 0.5
    # if energy_lag5 > 0.5:
    #     s["Energy (kWh)_lag5"] = 0.5
    
    return s

def tobit_model(label):
    df = load_data()
    df = df.drop(columns=['Charging Time (mins)', 'Parking Time (mins)','# Food', '# Shop & Service',
       '# Professional & Other Places', '# Nightlife Spot',
       '# Outdoors & Recreation', '# Arts & Entertainment',
       '# Travel & Transport', '# College & University', '# Event',
       '# Residence', 'Plugs_raw', 'Number of NEMA 5-20R_raw','Number of NEMA 5-20R', 'Label_0', 'Label_1', 'Label_2', 'Label_3',
       'Label_4', 'Label_5', 'Label_6', 'Label_7'])
    df["Fee (USD)"][df["Fee (USD)"]>0] = 1 
    df = df[df["Label"]==label]
    # df1 = pd.read_csv("data\createdDat\Censorship_scheem\latent1_26052021.csv")
    # df1 = df1[df1["Label"]==label]
    # df1["Date"]= pd.to_datetime(df1["Date"])
    # df1 = df1.set_index("Date")
    l = lags()
    testdata = df[df.Label == label]
    index = testdata[testdata["Energy (kWh)"]>0].index[0]
    testdata = testdata[index:] 
    index = df[df["Energy (kWh)"]>0].index[0]
    df= df[index:]

    scaler_eng = StandardScaler()

    # for i in range(1,13):
    #     testdata[f"Year_Month_{i}"] = 0
    #     testdata[f"Year_Month_{i}"][(testdata.index.month==i) & (testdata.index.day==1)] = 1
    #     testdata[f"Year_Month_{testdata.index[0].month}"]=1
    
    # testdata["Censored"] = False
    # testdata["Censored"][(testdata["Label"]==label) & (df1["Plugs"]==df1["Simultaneous use (daily max)_raw"])] = True


    #df["Censored"] = False
    #df["Censored"][(df["Label"]==label) & (df1["Plugs"]==df1["Simultaneous use (daily max)_raw"])] = True
    #df["Energy (kWh)"][df["Censored"]==True] = df["Energy (kWh)"][df["Censored"]==True] * 1.2

    testdata = testdata.apply(censor ,axis=1)
    testdata["Energy (kWh)"] = scaler_eng.fit_transform(np.array(testdata["Energy (kWh)"]).reshape(-1,1))
    #testdata["Fee (USD)"] = scaler_fee.fit_transform(np.array(testdata["Fee (USD)"]).reshape(-1,1))


    testdata = l.buildLaggedFeatures(testdata,["Energy (kWh)"])

    
    #print(testdata["Energy (kWh)"].describe())
    y = testdata["Energy (kWh)"]
    X = testdata
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    t = Tobit(X_train, 'Censored', list(X_train.drop(columns=["Energy (kWh)","Censored"]).columns), 'Energy (kWh)')

    regressor = Ridge(fit_intercept=False).fit(X_train.drop(columns=["Energy (kWh)","Censored"]), X_train["Energy (kWh)"])
    pred = regressor.predict(X_test.drop(columns=["Energy (kWh)","Censored"]))

    vars = regressor.coef_
    vars = np.append(vars,1)
    
    minimizer = t.minimize(vars)

    sd = minimizer['x'][-1]
    beta = minimizer['x'][:-1]


    pred_tobit = t.predict(np.array(X_test.drop(columns=["Energy (kWh)","Censored"])),beta)

    # rmse_tobit = np.sqrt(mean_squared_error(df["Energy (kWh)"][5:],scaler_eng.inverse_transform(pred_tobit)))
    # rmse_ridge = np.sqrt(mean_squared_error(df["Energy (kWh)"][5:],scaler_eng.inverse_transform(pred)))

    # cmap = plt.cm.bone
    # rmap = plt.cm.Reds
    # bmap = plt.cm.Blues

    # plt.plot(df["Energy (kWh)"][5:], label = "Original", alpha = 0.6, color="black")
    # plt.plot(testdata.index,scaler_eng.inverse_transform(pred_tobit), label = "Tobit", alpha = 0.6, color=rmap(0.9))
    # plt.plot(testdata.index,scaler_eng.inverse_transform(pred), label = "Ridge", alpha = 0.6, color=bmap(0.9))
    # plt.xlabel("Date")
    # plt.ylabel("Energy (kWh)")
    # plt.title(f"Cluster {label}")
    # plt.legend()
    # plt.show()

    predictions = pd.DataFrame(columns=["Tobit_pred","Ridge_pred", "Censored_energy"])
    #predictions = predictions.set_index(testdata.index)
    predictions["Tobit_pred"] = scaler_eng.inverse_transform(pred_tobit)
    predictions["Ridge_pred"] = scaler_eng.inverse_transform(pred)
    predictions["Censored_energy"] = y_test

    return predictions


#for normal censor


# for label in range(8):
#     tobit_model(label).to_csv(f"data\createdDat\Censorship_scheem\Tobit_0.5_test_censor_label{label}.csv")


def tobit_model_censorc(label):
    df = pd.read_csv("data\createdDat\Censorship_scheem\df_days_with_external_censoredB_27052021.csv")
    df = df.drop(columns=['Charging Time (mins)', 'Parking Time (mins)','# Food', '# Shop & Service',
       '# Professional & Other Places', '# Nightlife Spot',
       '# Outdoors & Recreation', '# Arts & Entertainment',
       '# Travel & Transport', '# College & University', '# Event',
       '# Residence', 'Plugs_raw', 'Number of NEMA 5-20R_raw','Number of NEMA 5-20R', 'Label_0', 'Label_1', 'Label_2', 'Label_3',
       'Label_4', 'Label_5', 'Label_6', 'Label_7'])
    df["Fee (USD)"][df["Fee (USD)"]>0] = 1 
   
    # df = df[df["Label"]==label]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    # df1 = pd.read_csv("data\createdDat\Censorship_scheem\latent1_26052021.csv")
    # df1 = df1[df1["Label"]==label]
    # df1["Date"]= pd.to_datetime(df1["Date"])
    # df1 = df1.set_index("Date")
    l = lags()
    testdata = df[df.Label == label]
    # index = testdata[testdata["Energy (kWh)"]>0].index[0]
    # testdata = testdata[index:] 
    # index = df[df["Energy (kWh)"]>0].index[0]
    # df= df[index:]

    scaler_eng = StandardScaler()
    #scaler_fee = StandardScaler()

    # for i in range(1,13):
    #     testdata[f"Year_Month_{i}"] = 0
    #     testdata[f"Year_Month_{i}"][(testdata.index.month==i) & (testdata.index.day==1)] = 1
    #     testdata[f"Year_Month_{testdata.index[0].month}"]=1
    
    # testdata["Censored"] = False
    # testdata["Censored"][(testdata["Label"]==label) & (df1["Plugs"]==df1["Simultaneous use (daily max)_raw"])] = True


    #df["Censored"] = False
    #df["Censored"][(df["Label"]==label) & (df1["Plugs"]==df1["Simultaneous use (daily max)_raw"])] = True
    #df["Energy (kWh)"][df["Censored"]==True] = df["Energy (kWh)"][df["Censored"]==True] * 1.2

    testdata["Energy (kWh)"] = scaler_eng.fit_transform(np.array(testdata["Energy (kWh)"]).reshape(-1,1))
    #testdata["Fee (USD)"] = scaler_fee.fit_transform(np.array(testdata["Fee (USD)"]).reshape(-1,1))

    testdata = l.buildLaggedFeatures(testdata,["Energy (kWh)"])

    y = testdata["Energy (kWh)"]
    X = testdata
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    #testdata = testdata.apply(censor, axis=1)

    t = Tobit(X_train, 'Censored', list(X_train.drop(columns=["Energy (kWh)","Censored"]).columns), 'Energy (kWh)')

    regressor = Ridge(fit_intercept=False).fit(X_train.drop(columns=["Energy (kWh)", "Censored"]), X_train["Energy (kWh)"])
    pred = regressor.predict(X_test.drop(columns=["Energy (kWh)","Censored"]))

    vars = regressor.coef_
    vars = np.append(vars,1)

    minimizer = t.minimize(vars)

    sd = minimizer['x'][-1]
    beta = minimizer['x'][:-1]

    pred_tobit = t.predict(np.array(X_test.drop(columns=["Energy (kWh)","Censored"])),beta)

    # rmse_tobit = np.sqrt(mean_squared_error(df["Energy (kWh)"][5:],scaler_eng.inverse_transform(pred_tobit)))
    # rmse_ridge = np.sqrt(mean_squared_error(df["Energy (kWh)"][5:],scaler_eng.inverse_transform(pred)))

    # cmap = plt.cm.bone
    # rmap = plt.cm.Reds
    # bmap = plt.cm.Blues

    # plt.plot(df["Energy (kWh)"][5:], label = "Original", alpha = 0.6, color="black")
    # plt.plot(testdata.index,scaler_eng.inverse_transform(pred_tobit), label = "Tobit", alpha = 0.6, color=rmap(0.9))
    # plt.plot(testdata.index,scaler_eng.inverse_transform(pred), label = "Ridge", alpha = 0.6, color=bmap(0.9))
    # plt.xlabel("Date")
    # plt.ylabel("Energy (kWh)")
    # plt.title(f"Cluster {label}")
    # plt.legend()
    # plt.show()

    predictions = pd.DataFrame(columns=["Tobit_pred","Ridge_pred", "Censored_energy"])
    #predictions = predictions.set_index(testdata.index)
    predictions["Tobit_pred"] = scaler_eng.inverse_transform(pred_tobit)
    predictions["Ridge_pred"] = scaler_eng.inverse_transform(pred)
    predictions["Censored_energy"] = testdata["Energy (kWh)"]

    return predictions



# for label in range(8):
#     tobit_model_censorc(label).to_csv(f"data\createdDat\Censorship_scheem\Tobit_c_censor_label{label}.csv")





def tobit_model_plugs(label):
    # df = pd.read_csv("data\createdDat\Censorship_scheem\df_censored.csv")
    # df = df.drop(columns=['Charging Time (mins)', 'Parking Time (mins)', 'Plugs_raw', 'Number of NEMA 5-20R_raw','Number of NEMA 5-20R', 'Label_0', 'Label_1', 'Label_2', 'Label_3',
    #    'Label_4', 'Label_5', 'Label_6', 'Label_7', 'Energy (kWh)_x', 'Fee (USD)_x', "Plugs"])
    
    # pois = ['# Professional & Other Places',
    #        '# Food', '# Shop & Service', '# Travel & Transport',
    #        '# Outdoors & Recreation', '# Arts & Entertainment', '# Nightlife Spot',
    #        '# Residence', '# College & University', '# Event']
    #df = pd.read_csv("data\createdDat\Censorship_scheem\df_censor_C_b2.csv")
    df = pd.read_csv("data\createdDat\Censorship_scheem\df_censor_b3.csv")
    to_drop =['Unnamed: 0','Charging Time (mins)', 'Parking Time (mins)', '# Food', 
    '# Shop & Service','# Professional & Other Places', '# Nightlife Spot','# Outdoors & Recreation', '# Arts & Entertainment',
    '# Travel & Transport', '# College & University', '# Event','# Residence', 'Plugs_raw', 'Number of NEMA 5-20R_raw', 'Label_0', 
    'Label_1', 'Label_2', 'Label_3','Label_4', 'Label_5', 'Label_6', 'Label_7', 'Simultaneous use (daily max)']
    df = df.drop(columns= to_drop)
    #df["Fee (USD)_y"][df["Fee (USD)_y"]>0] = 1
    # 
    df["Fee (USD)"][df["Fee (USD)"]>0] = 1  

    
    #df = df[df["Label"]==label]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    # df1 = pd.read_csv("data\createdDat\Censorship_scheem\latent1_26052021.csv")
    # df1 = df1[df1["Label"]==label]
    # df1["Date"]= pd.to_datetime(df1["Date"])
    # df1 = df1.set_index("Date")
    l = lags()
    df = df.fillna(0)
    scaler_poi = StandardScaler()
    #df[pois] = scaler_poi.fit_transform(df[pois])
    testdata = df[df.Label == label]

    # index = testdata[testdata["Energy (kWh)_y"]>0].index[0]
    # testdata = testdata[index:] 
    # index = df[df["Energy (kWh)_y"]>0].index[0]
    # df= df[index:]

    scaler_eng = StandardScaler()
    

    # for i in range(1,13):
    #     testdata[f"Year_Month_{i}"] = 0
    #     testdata[f"Year_Month_{i}"][(testdata.index.month==i) & (testdata.index.day==1)] = 1
    #     testdata[f"Year_Month_{testdata.index[0].month}"]=1
    
    # testdata["Censored"] = False
    # testdata["Censored"][(testdata["Label"]==label) & (df1["Plugs"]==df1["Simultaneous use (daily max)_raw"])] = True


    #df["Censored"] = False
    #df["Censored"][(df["Label"]==label) & (df1["Plugs"]==df1["Simultaneous use (daily max)_raw"])] = True
    #df["Energy (kWh)_y"][df["Censored"]==True] = df["Energy (kWh)_y"][df["Censored"]==True] * 1.2

    testdata["Energy (kWh)"] = scaler_eng.fit_transform(np.array(testdata["Energy (kWh)"]).reshape(-1,1))
    
    #testdata["Fee (USD)_y"] = scaler_fee.fit_transform(np.array(testdata["Fee (USD)_y"]).reshape(-1,1))

    testdata = l.buildLaggedFeatures(testdata,["Energy (kWh)"], lag=8)
    y = testdata["Energy (kWh)"]
    X = testdata
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    # testdata = testdata.apply(censor, axis=1)
    

    t = Tobit(X_train, 'Censored', list(X_train.drop(columns=["Energy (kWh)","Censored"]).columns), 'Energy (kWh)')

    regressor = Ridge(fit_intercept=False).fit(X_train.drop(columns=["Energy (kWh)", "Censored"]), X_train["Energy (kWh)"])
    pred = regressor.predict(X_test.drop(columns=["Energy (kWh)","Censored"]))

    vars = regressor.coef_
    vars = np.append(vars,1)
    #print(dict(zip(testdata.drop(columns=["Energy (kWh)_y","Censored"]).columns, np.round(vars[:-1],3))))

    minimizer = t.minimize(vars)
    
    #print(minimizer.message)


    sd = minimizer['x'][-1]
    beta = minimizer['x'][:-1]
    #print(dict(zip(testdata.drop(columns=["Energy (kWh)_y","Censored"]).columns, np.round(beta[:-1],3))))

    pred_tobit = t.predict(np.array(X_test.drop(columns=["Energy (kWh)","Censored"])),beta)

    #rmse_tobit = np.sqrt(mean_squared_error(df["Energy (kWh)_y"][5:],scaler_eng.inverse_transform(pred_tobit)))
    #rmse_ridge = np.sqrt(mean_squared_error(df["Energy (kWh)_y"][5:],scaler_eng.inverse_transform(pred)))

    # cmap = plt.cm.bone
    # rmap = plt.cm.Reds
    # bmap = plt.cm.Blues

    # plt.plot(df["Energy (kWh)"][5:], label = "Original", alpha = 0.6, color="black")
    # plt.plot(testdata.index,scaler_eng.inverse_transform(pred_tobit), label = "Tobit", alpha = 0.6, color=rmap(0.9))
    # plt.plot(testdata.index,scaler_eng.inverse_transform(pred), label = "Ridge", alpha = 0.6, color=bmap(0.9))
    # plt.xlabel("Date")
    # plt.ylabel("Energy (kWh)")
    # plt.title(f"Cluster {label}")
    # plt.legend()
    # plt.show()

    predictions = pd.DataFrame(columns=["Tobit_pred","Ridge_pred", "Censored_energy"])
    #predictions = predictions.set_index(testdata.index)
    predictions["Tobit_pred"] = scaler_eng.inverse_transform(pred_tobit)
    predictions["Ridge_pred"] = scaler_eng.inverse_transform(pred)
    predictions["Censored_energy"] = testdata["Energy (kWh)"]

    return predictions

for label in range(8):
  tobit_model_plugs(label).to_csv(f"data\createdDat\Censorship_scheem\Tobit_plug3_censor_label{label}.csv")
