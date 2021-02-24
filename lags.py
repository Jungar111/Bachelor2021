result = pd.DataFrame()


start = time.time()


for index, row in df16one.iterrows():

    dftemp = pd.DataFrame(row).T
    dftemp.insert(0, 'Slot', dftemp['StartSlot'])
    
    if dftemp.iloc[-1]['Start Date'] + dftemp.iloc[-1]['Charging Time (hh:mm:ss)'] > (dftemp.iloc[-1]['StartSlot'] + dt.timedelta(hours=2)):
        dftemp.insert(1, 'ChargeMin', (dftemp['StartSlot'] + dt.timedelta(hours=2)) - dftemp['Start Date'])
        dftemp.insert(2, 'TimeLeftCharge', dftemp['Charging Time (hh:mm:ss)'] - dftemp['ChargeMin'])
    else: 
        dftemp.insert(1, 'ChargeMin', dftemp.iloc[-1]['Charging Time (hh:mm:ss)'])
        dftemp.insert(2, 'TimeLeftCharge', dt.timedelta(seconds=0))

    if dftemp.iloc[-1]['Start Date'] + dftemp.iloc[-1]['Total Duration (hh:mm:ss)'] > (dftemp.iloc[-1]['StartSlot'] + dt.timedelta(hours=2)):
        dftemp.insert(3, 'ParkMin', (dftemp['StartSlot'] + dt.timedelta(hours=2)) - dftemp['Start Date'])
        dftemp.insert(4, 'TimeLeftPark', dftemp['Total Duration (hh:mm:ss)'] - dftemp['ChargeMin'])
    else: 
        dftemp.insert(3, 'ParkMin', dftemp.iloc[-1]['Total Duration (hh:mm:ss)'])
        dftemp.insert(4, 'TimeLeftPark', dt.timedelta(seconds=0))                      
 
    while dftemp.iloc[-1]['TimeLeftPark'] > dt.timedelta(seconds=0):
        dftemp2 = pd.DataFrame(dftemp.iloc[-1]).T
        dftemp2['Slot'] = dftemp2['Slot'] + dt.timedelta(hours=2)

        if dftemp2.iloc[-1]['TimeLeftCharge'] > dt.timedelta(hours=2):
            dftemp2['ChargeMin'] = dt.timedelta(hours=2)
            dftemp2['TimeLeftCharge'] = dftemp2['TimeLeftCharge'] - dt.timedelta(hours=2)
        else: 
            dftemp2['ChargeMin'] = dftemp2['TimeLeftCharge']
            dftemp2['TimeLeftCharge'] = dt.timedelta(seconds=0)
            
        if dftemp2.iloc[-1]['TimeLeftPark'] > dt.timedelta(hours=2):
            dftemp2['ParkMin'] = dt.timedelta(hours=2)
            dftemp2['TimeLeftPark'] = dftemp2['TimeLeftPark'] - dt.timedelta(hours=2)
        else: 
            dftemp2['ParkMin'] = dftemp2['TimeLeftPark']
            dftemp2['TimeLeftPark'] = dt.timedelta(seconds=0)        
            
            
        dftemp = pd.concat([dftemp, dftemp2])    

    final = dftemp.reset_index() 
    result = pd.concat([result, final])
    
end = time.time()
print(end - start)