import pandas as pd 
padata = pd.read_csv("data\PaloAltoData.csv")
padata2 = pd.read_csv("data\PaloAltoData2.csv")

#print(padata.head())
print(padata.isna().sum())

print(padata["MAC Address"].unique())
#print(padata2.shape)

# I dont see a huge difference in the two links? and still only 7 chargers??? and 5 MAC Adresses??
# the same latitudes and longitudes 
