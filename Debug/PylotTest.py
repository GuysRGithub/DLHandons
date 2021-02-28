import math
import random

import numpy as np
import matplotlib.pyplot as plt
counts = np.zeros(100)
fig, axes = plt.subplots(2, 2, sharex=True)
axes = axes.flatten()
x = np.arange(-4, 16, 0.1)
# Mangle subplots such that we can index them in a linear fashion rather than
# a 2D grid

plt.plot(x, np.sin(x))
plt.show()
