import pandas as pd
# ^^^ pyforest auto-imports - don't write above this line
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
from sklearn.metrics import mean_squared_error, r2_score




m = modelling()
df = m.df.drop(columns=["Start Date", "Charging Time (mins)", "Total Duration (mins)", "Port Number"])


cols_to_standardize = ['# Professional & Other Places', '# Food', '# Shop & Service',
       '# Travel & Transport', '# Outdoors & Recreation',
       '# Arts & Entertainment', '# Nightlife Spot', '# Residence',
       '# College & University', '# Event']


sc = StandardScaler()
stand_poi = sc.fit_transform(df[cols_to_standardize])


stand_poi = pd.DataFrame(stand_poi, index=df.index, columns=cols_to_standardize)


for i in cols_to_standardize:
    df[i] = stand_poi[i]


df["Label"] = df["Label"] + 1


#df = df.drop(columns=cols_to_standardize)


df = df.fillna(0)


X_train,X_test, X_val,y_train,y_test, y_val = m.ttsplit(df)


X_train


labels = df.Label.unique()


stackings_X_test = []
stackings_X_train = []
stackings_X_val = []
stackings_y_test = []
stackings_y_train = []
stackings_y_val = []
for l in labels:
    stackings_X_test.append(tf.ragged.constant(X_test[X_test.Label == l]))
    stackings_y_test.append((tf.ragged.constant(y_test[X_test.Label == l])))
    stackings_X_train.append((tf.ragged.constant(X_train[X_train.Label == l])))
    stackings_y_train.append((tf.ragged.constant(y_train[X_train.Label == l])))
    stackings_X_val.append((tf.ragged.constant(X_val[X_val.Label == l])))
    stackings_y_val.append((tf.ragged.constant(y_val[X_val.Label == l])))



df.isna().sum().sum()


for i in range(65):
    if X_train.isna().sum()[i] > 0:
        print(i)


X_train.columns[14]


X_train_stack = tf.ragged.stack(stackings_X_train).to_tensor()
X_test_stack =  tf.ragged.stack(stackings_X_test).to_tensor()
X_val_stack =tf.ragged.stack(stackings_X_val).to_tensor()
y_train_stack = tf.ragged.stack(stackings_y_train).to_tensor()
y_test_stack = tf.ragged.stack(stackings_y_test).to_tensor()
y_val_stack = tf.ragged.stack(stackings_y_val).to_tensor()





from datetime import datetime
logdir="logs/fit/" + datetime.now().strftime("get_ipython().run_line_magic("Y%m%d-%H%M%S")", "")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

m = Sequential()
m.add(Input(shape=(None,75), dtype=tf.float32))
m.add(LSTM(68, return_sequences = True, activation = "tanh"))

m.add(Dense(1, activation="relu"))

m.compile(optimizer="adam", loss='mse', metrics=[tf.keras.losses.MAPE])

history = m.fit(tf.transpose(X_train_stack, perm = [1,0,2]), tf.transpose(y_train_stack, perm = [1,0]), epochs=20 , batch_size=16, validation_data=(tf.transpose(X_val_stack, perm = [1,0,2]), tf.transpose(y_val_stack, perm = [1,0])), callbacks=tensorboard_callback)


y_pred = m.predict(tf.transpose(X_test_stack,perm=[1, 0, 2]))



tf.keras.utils.plot_model(m)


import tensorboard


get_ipython().run_line_magic("load_ext", " tensorboard")


get_ipython().run_line_magic("tensorboard", " --logdir logs --host localhost")


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


from sklearn.metrics  import mean_absolute_percentage_error, mean_absolute_error



for i in range(8):
    mse = np.sqrt(mean_squared_error(tf.transpose(y_pred, perm=[1,0,2])[i],tf.transpose(y_test_stack[i])))
    r2 = r2_score(tf.transpose(y_pred, perm=[1,0,2])[i],tf.transpose(y_test_stack[i]))
    MAPE = mean_absolute_percentage_error(tf.transpose(y_test_stack[i]),tf.transpose(y_pred, perm=[1,0,2])[i])
    MAE = mean_absolute_error(tf.transpose(y_test_stack[i]),tf.transpose(y_pred, perm=[1,0,2])[i])
    print(50*"-")
    print(f"Cluster: {i}\nR^2:{r2}\nRMSE:{mse}\nMAPE:{MAPE}\nMAE:{MAE}")
    print(50*"-")





for i in range(8):
    plt.scatter(y_test_stack[i][y_test_stack[i] get_ipython().getoutput("= 0], y_pred[0][i][tf.transpose(y_test_stack)[i] != 0])    ")
    plt.show()


df["Label"][df.Label.isin([1,2,3,6])] = 6





labels = list(df.Label.unique())


stackings_X_test = []
stackings_X_train = []
stackings_X_val = []
stackings_y_test = []
stackings_y_train = []
stackings_y_val = []
for l in labels:
    stackings_X_test.append(tf.ragged.constant(X_test[X_test.Label == l]))
    stackings_y_test.append((tf.ragged.constant(y_test[X_test.Label == l])))
    stackings_X_train.append((tf.ragged.constant(X_train[X_train.Label == l])))
    stackings_y_train.append((tf.ragged.constant(y_train[X_train.Label == l])))
    stackings_X_val.append((tf.ragged.constant(X_val[X_val.Label == l])))
    stackings_y_val.append((tf.ragged.constant(y_val[X_val.Label == l])))


X_train_stack = tf.ragged.stack(stackings_X_train).to_tensor()
X_test_stack =  tf.ragged.stack(stackings_X_test).to_tensor()
X_val_stack =tf.ragged.stack(stackings_X_val).to_tensor()
y_train_stack = tf.ragged.stack(stackings_y_train).to_tensor()
y_test_stack = tf.ragged.stack(stackings_y_test).to_tensor()
y_val_stack = tf.ragged.stack(stackings_y_val).to_tensor()


m = Sequential()
m.add(Input(shape=(None,75), dtype=tf.float32))
m.add(LSTM(68, return_sequences = True, activation = "tanh"))
m.add(Dropout(0.5))
m.add(LSTM(32, return_sequences = True, activation = "tanh"))
m.add(Dense(1, activation="relu"))

m.compile(optimizer="adam", loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = m.fit(tf.transpose(X_train_stack, perm = [1,0,2]), tf.transpose(y_train_stack, perm = [1,0]), epochs=20 , batch_size=16, validation_data=(tf.transpose(X_val_stack, perm = [1,0,2]), tf.transpose(y_val_stack, perm = [1,0])))


y_pred = m.predict(tf.transpose(X_test_stack,perm=[1, 0, 2]))


for i in range(5):
    mse = np.sqrt(mean_squared_error(tf.transpose(y_pred, perm=[1,0,2])[i],tf.transpose(y_test_stack[i])))
    r2 = r2_score(tf.transpose(y_pred, perm=[1,0,2])[i],tf.transpose(y_test_stack[i]))
    print(50*"-")
    print(f"Cluster: {labels[i]}\nR^2:{r2}\nRMSE:{mse}")
    print(50*"-")


X_train.shape


X_val.shape


m = Sequential()
m.add(Input(shape=(75,1), dtype=tf.float32))
m.add(LSTM(68, return_sequences = True, activation = "tanh"))
m.add(Dropout(0.5))
m.add(LSTM(32, return_sequences = True, activation = "tanh"))
m.add(Dense(1, activation="relu"))

m.compile(optimizer="adam", loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = m.fit(np.array(X_train).reshape(12445,75,1), y_train, epochs=20 , batch_size=256, validation_data=(np.array(X_val).reshape(1383,75,1), y_val))
