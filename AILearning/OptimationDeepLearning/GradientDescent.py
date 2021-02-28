from d2l import AllDeepLearning as d2l
from mxnet import nd
import numpy as np


# def f(x):
#     return x**2

#
# def gradf(x):
#     return 2 * x

#
# def gd(eta):
#     x = 10
#     results = [x]
#     for i in range(10):
#         x -= eta * gradf(x)
#         results.append(x)
#     print('epoch 10, x:', x)
#     return results


def show_trace(res):
    n = max(abs(min(res)), abs(max(res)))
    f_line = np.arange(-n, n, 0.01)
    d2l.set_figsize((3.5, 2.5))
    d2l.plot([f_line, res], [[f(x) for x in f_line], [f(x) for x in res]],
             'x', 'f(x)', fmts=['-', '-o'])


c = 0.15 * np.pi
# def f(x):
#     return x * np.cos(c * x)
#
#
# def gradf(x):
#     return np.cos(c * x) - c * x * np.sin(c * x)


def train_2d(trainer, steps=20):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    print(results)
    for i in range(steps):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    print(results)
    return results


def show_trace_2d(f, results):
    d2l.set_figsize((3.5, 2.5))
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1),
                         np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')


# def f(x1, x2):
#     return x1 ** 2 + 2 * x2 ** 2
#
#
# def gradf(x1, x2):
#     return (2 * x1, 4 * x2)


# def gd(x1, x2, s1, s2):
#     (g1, g2) = gradf(x1, x2)
#     return (x1 - eta * g1, x2 - eta * g2, 0, 0)

c = 0.5


def f(x):
    return  c * np.cosh(c * x)


def gradf(x):
    return c**2 * np.cosh((c * x))


def hessf(x):
    return  c**2 * np.cosh((c * x))


def newton(eta=1):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * gradf(x) / hessf(x)
        results.append(x)
    print('epoch 10, x', x)
    return results


show_trace(newton())
d2l.plt.show()