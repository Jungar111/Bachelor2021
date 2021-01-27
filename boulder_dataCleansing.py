import pandas as pd
import numpy as np
import os
os.getcwd()
import matplotlib as plt

#Importing data
boulder_df = pd.read_csv('./data/Boulder/Electric_Vehicle_Charging_Station_Energy_Consumption.csv')
boulder_df.head()
boulder_df.info()

#All variables 
boulder_df.columns.values.tolist()
#There's a description of all variables in ev_datadictionary.csv (NB! TransactionDateExtract & ObjectID is not mentioned)
boulder_df['TransactionDateExtract'].head()
boulder_df['Transaction_Date'].head() #Not the same? 
#NB! There's a weird lag in TransactionDateExtract it's one date behind. I reckon it's more reasonable to 
#create a new date extract based on Transaction_Date 


boulder_df['Transaction_Start_Time'].head()


#Checking for NaNs 
boulder_df.isna().sum() #There's only NaNs in 3 variables
boulder_df.describe()


