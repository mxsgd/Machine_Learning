
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
x = np.arange(10)
y = np.arange(10)
x, y = np.meshgrid(x, y)
z = -(x**2+y**3)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=True)
plt.show()