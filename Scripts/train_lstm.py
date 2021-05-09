import sys
sys.path.append(".")
from DataPrep.load_data import load_data
from Modelling import modelling
import keras
from keras import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Reshape
import pyforest
import torch
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

"""
Script for training LSTM model on DTU SSH
"""

#Load data
df = load_data()

#Standardize
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

def standardize(v):
    return (v - v.mean())/v.std()

for l in df.Label.unique():
    df["Energy (kWh)"][df.Label == l] = standardize(df["Energy (kWh)"][df.Label == l])


