import numpy as np
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt
import time
import re
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


df = pd.read_csv("Notebooks/df_04052021.csv")
p2 = pd.read_csv('Notebooks/p2_04052021.csv')
p2 = p2.rename(columns = {'Unnamed: 0': 'Date'})
p2['Date'] = pd.to_datetime(p2['Date'])
p2 = p2.set_index('Date')

def used_plugs(cluster, df):
    test = df.copy()
    test = test.drop(columns = 'Unnamed: 0')
    test = test[test['Label'] == cluster]

    start_time = time.time()
    dfsim = pd.DataFrame()

    for index, row in tqdm(test.iterrows(), total=test.shape[0]):


    #for index, row in test.iterrows():
        df_onecharge = pd.DataFrame(index = pd.date_range(start=row['Start Date'], end=row['End Date'], freq='Min'))
        
        dfsim = pd.concat([dfsim, df_onecharge])
    print("This bastard took", (time.time() - start_time)/60, "minutes to run...")


    dfsim2 = dfsim[~dfsim.index.duplicated(keep='first')]

    start_time = time.time()
    for index, row in tqdm(test.iterrows(), total=test.shape[0]):
        one = pd.DataFrame(index = pd.date_range(start=row['Start Date'], end=row['End Date'], freq='Min'))
        one[index] = 1
        dfsim2 = dfsim2.merge(one, how = 'left', left_index = True, right_index = True)    
        #s = row['Start Date']
        #e = row['End Date']
        #dfsim2[f'Charge {index}'] = 0
        #dfsim2[f'Charge {index}'].loc[s:e] = 1
    print("This bastard took", (time.time() - start_time)/60, "minutes to run...")
    dfsim2['Active plugs'] = dfsim2.sum(axis = 1)
    dfsim2.to_csv(f"dfsim2_{cluster}.csv")
    
for i in df.Label.unique():
    used_plugs(i, df)



