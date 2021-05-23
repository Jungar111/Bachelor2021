import sys
sys.path.append(".")
from DataPrep.load_data import load_data
import tensorflow.keras.backend as K
from tensorflow.keras.applications import *
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from tensorflow.keras.models import *
from tensorflow import keras
from tensorflow.keras import regularizers
import pyforest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import kerastuner as kt
cmap = plt.cm.bone
rmap = plt.cm.Reds


"""
LSTM hp tuning @ DTU ssh
"""

def create_data(X,y, pred_len, time_slot):

    Xs, ys = [], []

    for i in range(int(len(X) - pred_len - time_slot + 1)):
        X_start = i
        X_end = X_start + time_slot
        y_start = X_end
        y_end = y_start + pred_len
        v = X.iloc[X_start:X_end].values
        Xs.append(v)
        ys.append(y[y_start:y_end])

    return np.array(Xs), np.array(ys)

def standardize(t_train, t_test, t_val, v_train, v_test, v_val, *columns):
    t_train = np.array(t_train)
    t_test = np.array(t_test)
    t_val = np.array(t_val)
    
    v_train = np.array(v_train)
    v_test = np.array(v_test)
    v_val = np.array(v_val)
    
    statistics = {}
    
    for c in columns:
        first = t_train[0,:,c]
        rest = t_train[1:,-1:,c]
        series = np.append(first,rest)
        
        mu = series.mean()
        std = series.std()
        
        statistics[c] = (mu, std)
        
        t_train[:,:,c] = (t_train[:,:,c] - mu) / std
        t_test[:,:,c] = (t_test[:,:,c] - mu) / std
        t_val[:,:,c] = (t_val[:,:,c] - mu) / std
        
        if c == 3:
            v_train = ((v_train - mu) / std)
            v_test = ((v_test - mu) / std)
            v_val = ((v_val - mu) / std)
            
            
    return (tf.convert_to_tensor(t_train), tf.convert_to_tensor(t_test), tf.convert_to_tensor(t_val),
           tf.convert_to_tensor(v_train), tf.convert_to_tensor(v_test), tf.convert_to_tensor(v_val), statistics)
    
    
    

def get_until_not_zero(df):
    index = df[df["Energy (kWh)"] != 0].index[0]
    df = df.loc[index:]
    return df

def get_data(cluster, *columns):
    df_temporal = df.drop(columns=['# Professional & Other Places',
           '# Food', '# Shop & Service', '# Travel & Transport',
           '# Outdoors & Recreation', '# Arts & Entertainment', '# Nightlife Spot',
           '# Residence', '# College & University', '# Event', 'Label_0',
           'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5', 'Label_6', 'Label_7'])

    df_0 = df_temporal[df_temporal.Label == cluster]
    df_0 = get_until_not_zero(df_0)
    df_0 = df_0.sort_index()
    
    X = df_0
    y = df_0["Energy (kWh)"]

    X, y = create_data(X,y, 7, 150)
    
    split = [int(X.shape[0] * 0.9), int(X.shape[0] * 0.05), X.shape[0] - (int(X.shape[0] * 0.9) + int(X.shape[0] * 0.05))]
    
    #split = [X.shape[0] - (int(X.shape[0]*0.7) + int(X.shape[0]*0.2)), int(X.shape[0]*0.7), int(X.shape[0]*0.2)]
    


    X_train, X_val, X_test = tf.split(X, split)

    y_train, y_val, y_test = tf.split(y, split)

    X_train, X_test, X_val, y_train, y_test, y_val, statistics = standardize(X_train, X_test, X_val, y_train, y_test, y_val, 1,2,3,4)

    return (X_train, X_test, X_val, y_train, y_test, y_val, statistics)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def build_model(hp):
    inputs_lstm = Input(shape=(150,56))

    for i in range(hp.Int("LSTM_layers",1,2)):
        if i == 0:
            x = LSTM(hp.Int(f"LSTM_{i}",1,55), return_sequences = True, activation = "tanh")(inputs_lstm)
            x = Dropout(hp.Float(f'lstm{i}_dropout', min_value=0, max_value=0.9, step=0.1))(x)
        else:
            x = LSTM(hp.Int(f"LSTM_{i}",1,55), return_sequences = True, activation = "tanh")(x)
            x = Dropout(hp.Float(f'lstm{i}_dropout', min_value=0, max_value=0.9, step=0.1))(x)

    
    x = LSTM(hp.Int(f"LSTM_last",1,10), return_sequences = False, activation = "tanh")(x)
    x = Dropout(hp.Float(f'Dense_dropout', min_value=0, max_value=0.9, step=0.1))(x)

    outputs_lstm = Dense(7, kernel_regularizer=regularizers.l2(hp.Float(f'lambda_l2', min_value=0, max_value=0.005, step=0.0001)))(x)


    model = keras.Model(inputs=inputs_lstm, outputs=outputs_lstm, name="LSTM_model")
    opt = keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))

    model.compile(optimizer=opt, loss='mse', metrics=[rmse])

    return model



df = load_data()
df = df.fillna(0)

"""
1 : Charging Time
2 : Parking Time
3 : Energy
4 : Fee
"""

for i in range(8):
    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective("val_rmse", direction="min"),
        max_trials = 300,
        directory='Modelling/HyperParameterTuning',
        project_name=f'LSTM_label_{i}')
    X_train, X_test, X_val, y_train, y_test, y_val, standardize_statistics = get_data(i, 1,2,3,4)
    tuner.search(X_train, y_train,epochs=256, batch_size = 128, validation_data=(X_val, y_val), callbacks = [keras.callbacks.EarlyStopping(monitor = "val_rmse", patience = 20)])
    