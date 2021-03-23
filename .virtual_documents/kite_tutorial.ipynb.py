# Run me!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


x = np.linspace(-np.pi, np.pi, 50)
y = np.sin(x)
plt.plot(x,y)




# Run me!
url = 'https://kite.com/kite-public/iris.csv'
df = pd.read_csv(url)
df.head()


# Put code in me

plt.scatter(df['sepal_length'], df['sepal_width'])
