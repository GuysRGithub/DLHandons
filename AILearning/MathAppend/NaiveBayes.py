import d2l.AllDeepLearning as d2l
import math
from mxnet import gluon, nd
from mxnet import nd as np


def transform(data, label):
    return nd.floor(data.astype('float32') / 128).squeeze(axis=-1), label


mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)


images, labels = mnist_train[10:38]

X, Y = mnist_train[:]
n_y = nd.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
n_x = nd.zeros((10, 28, 28))

for y in range(10):
    n_x[y] = nd.array(X.asnumpy()[Y == y].sum(axis=0))

P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

X, Y = mnist_train[:]  # All training examples
n_y = np.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

# d2l.show_images(P_xy, 2, 5)
# d2l.plt.show()


def bayes_pred_log(x):
    """
    y=arg_max_y∑i=1->d(logPxy[xi,y]+logPy[y]).
    :param x: data
    :return: p(x/y) bayes pred
    """
    log_P_xy = np.log(P_xy)
    log_P_xy_neg = np.log(1 - P_xy)
    log_P_y = np.log(P_y)
    x = np.expand_dims(x, axis=0)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)
    return p_xy + log_P_y


image, label = mnist_train[2]
py = bayes_pred_log(image)
print(py.argmax(axis=0).asscalar() == int(label))
