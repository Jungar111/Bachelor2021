from Data_cleaning import clean_paloalto
import pandas as pd

c = clean_paloalto()
data = c.clean_data()

data.to_csv("clean_data.csv")
