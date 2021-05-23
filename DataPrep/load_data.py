import sys
sys.path.append(".")
from pathlib import Path

import pandas as pd

def load_data():
    p = Path('data', 'createdDat', 'AlmostClean.csv')
    df = pd.read_csv(p)
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    #df = df.drop(columns=['Charging Time (mins)', 'Parking Time (mins)'])
    return df


if __name__ == "__main__":
    print(load_data())
    