import matplotlib.pyplot as plt
import numpy as np
x = np.arange(5)
y = (9-4)*x**2+(3-5)*x+(3-6)
g = (np.e**x)/(np.e**x+1)
plt.plot(x, y, 'g')
plt.plot(x, g, 'r')
plt.show()
print(np.e)