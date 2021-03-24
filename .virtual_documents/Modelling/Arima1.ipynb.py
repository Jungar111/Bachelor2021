import matplotlib.pyplot as plt
# ^^^ pyforest auto-imports - don't write above this line



import sys
sys.path.append("..")
get_ipython().run_line_magic("cd", " ..")


import pyforest
import statsmodels.api as sm
from DataPrep.ImportData import importer
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
from sklearn.metrics import r2_score, mean_squared_error


df = importer().Import()


df.index = df["Start Date"]
df = df.drop(columns = ["Start Date"])


df


df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)



df.head()


#df = df.sort_values("Start Date")



labels = df.Label.unique()


labels


y = df["Energy (kWh)"]
X = df.drop(columns = ["Energy (kWh)"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)


fig, ax = plt.subplots(4,2, figsize=(10,20))


ax[0][0].scatter(y_train[X_train.Label == labels[0]].index,y_train[X_train.Label == labels[0]])
ax[0][0].title.set_text("Cluster 0")

ax[1][0].scatter(y_train[X_train.Label == labels[1]].index,y_train[X_train.Label == labels[1]])
ax[1][0].title.set_text("Cluster 1")

ax[2][0].scatter(y_train[X_train.Label == labels[2]].index,y_train[X_train.Label == labels[2]])
ax[2][0].title.set_text("Cluster 2")

ax[3][0].scatter(y_train[X_train.Label == labels[3]].index,y_train[X_train.Label == labels[3]])
ax[3][0].title.set_text("Cluster 3")

ax[0][1].scatter(y_train[X_train.Label == labels[4]].index,y_train[X_train.Label == labels[4]])
ax[0][1].title.set_text("Cluster 4")

ax[1][1].scatter(y_train[X_train.Label == labels[5]].index,y_train[X_train.Label == labels[5]])
ax[1][1].title.set_text("Cluster 5")

ax[2][1].scatter(y_train[X_train.Label == labels[6]].index,y_train[X_train.Label == labels[6]])
ax[2][1].title.set_text("Cluster 6")

ax[3][1].scatter(y_train[X_train.Label == labels[7]].index,y_train[X_train.Label == labels[7]])
ax[3][1].title.set_text("Cluster 7")

plt.show()


linear_model = ARIMA(y_train, order = (5,0,0), trend='n', enforce_invertibility=False, enforce_stationarity=False)


model_fit = linear_model.fit(gls=True)


print(model_fit.summary())


y_pred = model_fit.forecast(steps=len(y_test))


plt.plot(y_pred.index, y_pred.values)
plt.show()


plt.plot(y_pred.index, y_test, label="Test set")
plt.plot(y_pred.index, y_pred.values, label="Predictions")
plt.title("Predictions")
plt.legend()
plt.show()


plt.scatter(y_test, y_pred.values)
plt.title("Actual vs. predicted")
plt.show()


r2_score(y_test, y_pred)


model_fit.specification


plt.scatter(y_train,model_fit.predict())
plt.title("Training Data vs. Predictions")
plt.show()


plt.plot(y_train)
plt.plot(model_fit.predict())
plt.title("Training Data vs. Predictions")
plt.show()


r2_score(y_train, model_fit.predict())


mean_squared_error(y_train, model_fit.predict())


years = df.index.year.unique()
df["Energy (kWh)"]





import warnings
warnings.filterwarnings("ignore")



for year in years:
    if year < 2019:
        y_train = df["Energy (kWh)"][df.index.year <= year]
        linear_model = ARIMA(y_train, order = (5,0,0), trend='n', enforce_invertibility=False, enforce_stationarity=False)
        model_fit = linear_model.fit(gls=True)
        y_test = df["Energy (kWh)"][df.index.year == year + 1]
        y_pred = model_fit.forecast(steps=len(y_test))
        y_pred_in = model_fit.predict()
        print(40*"-")
        print(f"For training up until year: {year} \nOut-sample \nMSE = {mean_squared_error(y_test, y_pred)}\nr^2 = {r2_score(y_test, y_pred)} \n")
        print(f"in-sample \nMSE = {mean_squared_error(y_train, y_pred_in)}\nr^2 = {r2_score(y_train, y_pred_in)}")
        print(40*"-")

















stepwise_model = auto_arima(y_train[X_train.Label == labels[0]], 
                            start_p=5, 
                            start_q=2,
                            max_p=10, 
                            max_q=10, 
                            m=12,
                            start_P=0, 
                            seasonal=True,
                            d=1, 
                            D=1, 
                            trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True
                           )


stepwise_model.fit(y_train[X_train.Label == labels[0]])


print(f'AR-params:\n{stepwise_model.arparams()} \n\nMA-params:\n{stepwise_model.maparams()}')


y_pred = stepwise_model.predict(n_periods=len(y_test[X_test.Label == labels[0]]))


plt.scatter(y_pred, y_test[X_test.Label == labels[0]])
plt.show()


r2_score(y_test[X_test.Label == labels[0]],y_pred)
