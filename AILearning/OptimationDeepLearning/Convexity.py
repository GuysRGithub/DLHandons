from d2l import AllDeepLearning as d2l
from mxnet import nd
from mpl_toolkits import mplot3d
import numpy as np


def f(x):
    return 0.5 * x**2


def g(x):
    return nd.cos(np.pi * x)


def h(x):
    return nd.exp(0.5 * x)


x, y =np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101), indexing='ij')

z = x**2 + 0.5 * np.cos(2 * np.pi * y)

d2l.set_figsize((6, 4))
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.contour(x, y, z, offset=-1)
ax.set_zlim(-1, 1.5)

for func in [d2l.plt.xticks, d2l.plt.yticks, ax.set_zticks]:
    func([-1, 0, 1])

d2l.plt.show()