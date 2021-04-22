import sys
sys.path.append(".")

import pandas as pd

def load_data():
    
    df = pd.read_csv("data/createdDat/AlmostClean.csv")
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    return df
