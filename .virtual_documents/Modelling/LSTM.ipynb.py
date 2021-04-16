import sys
sys.path.append("..")
get_ipython().run_line_magic("cd", " ..")


from DataPrep.ImportData import importer
from Modelling import modelling
import keras
from keras import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Reshape
import pyforest
import torch
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error



m = modelling()
df = m.df.drop(columns=["Charging Time (mins)", "Total Duration (mins)", "Port Number"])





cols_to_standardize = ['# Professional & Other Places', '# Food', '# Shop & Service',
       '# Travel & Transport', '# Outdoors & Recreation',
       '# Arts & Entertainment', '# Nightlife Spot', '# Residence',
       '# College & University', '# Event']

sc = StandardScaler()
stand_poi = sc.fit_transform(df[cols_to_standardize])
stand_poi = pd.DataFrame(stand_poi, index=df.index, columns=cols_to_standardize)
for i in cols_to_standardize:
    df[i] = stand_poi[i]


df = df.fillna(0)


df.head()


df = df.set_index("Start Date")
df.index = df.index.to_period("D")


labels = df.Label.unique()


df_reindex = pd.DataFrame()
idx = pd.period_range(min(df.index), max(df.index))
for i in labels:
    filling = df[df.Label == i].reindex(idx, fill_value = 0)
    filling.Label = i
    df_reindex = df_reindex.append(filling)


df = df_reindex.sort_index()


df.head()


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(int((len(X) - time_steps)/8)):
        v = X.iloc[i*8:i*8+time_steps].values
        Xs.append(v)
        ys.append(y[i*8 + time_steps + 1])
    return tf.convert_to_tensor(np.array(Xs)), tf.convert_to_tensor(np.array(ys))


df.head()


def ttsplit(df,target="Energy (kWh)",shuffle=True):
    X = df
    y = df[target]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42,shuffle=shuffle)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.10, random_state=42,shuffle=shuffle)

    return X_train,X_test, X_val,y_train,y_test, y_val


X_train, X_test, X_val, y_train, y_test, y_val = ttsplit(df, shuffle = False)


X_train,y_train = create_dataset(X_train, y_train, time_steps = 182*8)


X_test,y_test = create_dataset(X_test, y_test, time_steps = 182*8)


X_val,y_val = create_dataset(X_val, y_val, time_steps = 182*8)


X_train.shape


X_train[:,:1, 1]


y_train


inputs = Input(shape=(None,80))
x = LSTM(80, return_sequences = True, activation = "tanh")(inputs)
x = Dropout(0.1)(x)
x = Dense(8, activation="relu")(x)
outputs = Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_model")
#opt = keras.optimizers.Adam(learning_rate=1*10**(-4))
model.compile(optimizer="Adam", loss='mse', metrics=["mae"])

history = model.fit(X_train, y_train, epochs=5, batch_size=1, validation_data = (X_val,y_val))
