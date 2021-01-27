import pandas as pd 
padata = pd.read_csv("data\PaloAltoData.csv")
padata2 = pd.read_csv("data\PaloAltoData2.csv")
cpdata = pd.read_csv("data\ChargePoint Data 2017Q4.csv")

#print(padata.head())
print(cpdata.isna().sum())

print(len(cpdata["Longitude"].unique()))
#print(cpdata.shape)

# I dont see a huge difference in the two links? and still only 7 chargers??? and 5 MAC Adresses??
# the same latitudes and longitudes 


# found source of second link, and there seems to be much more data here. However seems to be some descrepencies between how 
# many chargers there are depending on Long/lat, Station Name and MAC Address? Which sould we use?
# Otherwise there dont seem to be problems with nans in the "important" categories
