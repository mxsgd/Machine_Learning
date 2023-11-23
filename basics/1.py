import csv
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../2/data2.csv")
data_array = data.to_numpy()

x = data_array[:, 1]
y = data_array[:, 2]

plt.plot(x, y, "g.")
plt.show()