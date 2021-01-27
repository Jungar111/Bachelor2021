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

#Turning it into datetime
boulder_df['TransactionDateExtract'].head()
boulder_df['Transaction_Date'].head() #Not the same? 
#NB! There's a weird lag in TransactionDateExtract it's one date behind. There's something fishy and undescribed in TransactionDateExtract (when they collecte the data?)
boulder_df['Transaction_Date'] = pd.to_datetime(boulder_df['Transaction_Date'], format = '%Y/%m/%d').dt.date
boulder_df['Transaction_Start_Time'] = pd.to_datetime(boulder_df['Transaction_Start_Time']).dt.time

#Checking for NaNs 
boulder_df.isna().sum() #There's only NaNs in 3 variables (But some other problems with "-" etc... )
boulder_df.describe()
#I'm not completely sure that we are interested in those variables containing NaNs... 
# Energy_kWh_ - "The amount of energy that has been dispensed by the charging stations on the particular listed date. Energy is measured in kilowatt hours (kWh)"
#Demand could perhaps just be modelled as time spent? 

boulder_df['Charging_Time__minutes_'].head() #Aha! A "-" instead of a "regular" NaN
boulder_df['Charging_Time__minutes_'].str.isalnum().sum()


#SOME HARSH FILTERING 
#To do: write a smart function that removes all rows containing NaNs, " ", or a non-alphanumeric character 
boulder_df.replace(to_replace = '-', value = np.NaN, inplace = True)
boulder_df.isna().sum()

boulder_df[boulder_df['Charging_Time__minutes_'].str.contains("-")]


teststr = 'abc'
teststr.isalnum()

boulder_df[boulder_df == '-']
