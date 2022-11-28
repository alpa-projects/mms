import os

import numpy as np
import matplotlib.pyplot as plt

shape, mode = 3., 2.
s = (np.random.pareto(shape, 1000) + 1) * mode

count, bins, _ = plt.hist(s, 100, density=True)
fit = shape * mode ** shape / bins ** (shape + 1)
plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
plt.show()
fig = plt.gcf()
figure_size = (8, 4)
fig.set_size_inches(figure_size)
fig.savefig("test.png", bbox_inches='tight')