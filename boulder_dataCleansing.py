import pandas as pd
import numpy as np
import os
os.getcwd()
import matplotlib as plt

#Importing data
#C:\Users\User\OneDrive - Danmarks Tekniske Universitet\SAS_030919\6. Semester\BSc\Bachelor2021\data\Boulder\Electric_Vehicle_Charging_Station_Energy_Consumption.csv
boulder_df = pd.read_csv('./data/Boulder/Electric_Vehicle_Charging_Station_Energy_Consumption.csv')
boulder_df.head()
boulder_df.info()