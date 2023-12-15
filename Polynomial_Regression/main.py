import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv(
    "data6.tsv",
    sep="\t",)
data_array = data.to_numpy()

reg = linear_model.LinearRegression()
Ridge = linear_model.Ridge(alpha=1)

x = data_array[:, 0].reshape(-1, 1)
y = data_array[:, 1]

reg.fit(x,y)

x_plot = np.linspace(min(x), max(x), 1000).reshape(-1, 1)

reg_quad = make_pipeline(PolynomialFeatures(2), Ridge)
reg_quad.fit(x, y)

reg_5 = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
reg_5.fit(x, y)

y_quad = reg_quad.predict(x_plot)
y_pred = reg.predict(x_plot)
y_5 = reg_5.predict(x_plot)

plt.scatter(x,y, color="black")
plt.plot(x_plot, y_pred, color="blue" )
plt.plot(x_plot, y_quad , color="red" )
plt.plot(x_plot, y_5 , color="green" )


plt.show()