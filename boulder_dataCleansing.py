import pandas as pd
import numpy as np
import os
os.getcwd()
#import matplotlib as plt
import matplotlib.pyplot as plt 


#Importing data
boulder_df = pd.read_csv('./data/Boulder/Electric_Vehicle_Charging_Station_Energy_Consumption.csv')
boulder_df.head()
boulder_df.info()
boulder_df.shape()

#All variables 
boulder_df.columns.values.tolist()
boulder_df['TransactionDateExtract'].head()
boulder_df['Transaction_Date'].head() #Not the same? 
#NB! There's a weird lag in TransactionDateExtract it's one date behind. There's something fishy and undescribed in TransactionDateExtract (when they collecte the data?)

boulder_df['Transaction_Date'] = pd.to_datetime(boulder_df['Transaction_Date']) #, format = '%Y/%m/%d'
boulder_df['Transaction_Start_Time'] = pd.to_datetime(boulder_df['Transaction_Start_Time'])
#boulder_df['Transaction_Date'] = pd.Series([val.date() for val in boulder_df['Transaction_Date']])
#boulder_df['Transaction_Start_Time'] = pd.Series([val.time() for val in boulder_df['Transaction_Start_Time']])


#boulder_df['Transaction_Date'] = pd.to_datetime(boulder_df['Transaction_Date'], format = '%Y/%m/%d')#.dt.date
#boulder_df['Transaction_Start_Time'] = pd.to_datetime(boulder_df['Transaction_Start_Time'])#.dt.time

#Checking for NaNs 
boulder_df = boulder_df.replace('-', ' ', regex=True)
#boulder_df.replace(' ', np.NaN, regex=True)
boulder_df = boulder_df.replace(r'^\s*$', np.nan, regex=True)
boulder_df['Charging_Time__minutes_'].head() #Aha! A "-" instead of a "regular" NaN
boulder_df['Charging_Time__minutes_'].str.isalnum().sum()
boulder_df.isna().sum()

#SOME HARSH FILTERING 
#To do: write a smart function that removes all rows containing NaNs, " ", or a non-alphanumeric character 
boulder_df['Charging_Time__minutes_'] = boulder_df['Charging_Time__minutes_'].replace(',', '', regex=True)

boulder_df['Charging_Time__minutes_'] = pd.to_numeric(boulder_df['Charging_Time__minutes_'])
boulder_df.info()


#### VISUALIZATIONS ####
#Transactions per date
date_count = pd.DataFrame(boulder_df['Transaction_Date'].value_counts().reset_index())
date_count.columns = ['date', 'count']
date_count.sort_values('date', ascending = True)

plt.bar(date_count['date'], date_count['count'], width = 1.0)
plt.xticks(rotation='vertical')
plt.ylabel('Number of Transactions')
plt.title('Transactions per date')
plt.show()

#Transactions per time interval 
df_time = boulder_df.copy()
df_time['hour'] = df_time['Transaction_Start_Time'].dt.hour
df_time[['Transaction_Start_Time', 'hour']].head()

hour_count = pd.DataFrame(df_time['hour'].value_counts().reset_index())
hour_count.columns = ['hour', 'count']
hour_count.sort_values('hour', ascending = True)

plt.bar(hour_count['hour'], hour_count['count'])
plt.ylabel('Number of Transactions')
plt.xlabel('Hours since midnight')
plt.title('Transactions per hour')
plt.show()

#Transactions per station 
station_count = pd.DataFrame(boulder_df['Station_Name'].value_counts().reset_index())
station_count.columns = ['station', 'count']
station_count

plt.bar(station_count['station'], station_count['count'])
plt.xticks(rotation='vertical')
plt.ylabel('Number of Transactions')
plt.title('Transactions per EVSE')
plt.show()


##### VERY BASIC CORRELATION CHECK 
boulder_df.corr()
plt.plot(boulder_df['Charging_Time__minutes_'], boulder_df['Energy__kWh_'], '.')
plt.show()

print(boulder_df['Port_Type'].unique()) #definatley not fast charging 