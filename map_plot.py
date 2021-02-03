from Palo_Alto_Initial import clean_paloalto
import matplotlib.pyplot as plt

c = clean_paloalto()
data =  c.clean_data()
print(data.columns)

print(max(data.Longitude), max(data.Latitude))
print(min(data.Longitude), min(data.Latitude))

plt.scatter(data.Longitude, data.Latitude, s = 0.1)
plt.show()