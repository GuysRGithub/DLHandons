from mxnet import nd
from mxnet.gluon import nn


def pool2d(X, pool_size, mode="max"):
    pool_h, pool_w = pool_size
    Y = nd.zeros((X.shape[0] - pool_h + 1, X.shape[1] - pool_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i: i + pool_h, j: j + pool_w].max()
            elif mode == "avg":
                Y[i, j] = X[i: i + pool_h, j: j+ pool_w].mean()
    return Y


X = nd.arange(16).reshape((1, 1, 4, 4))
pool = nn.MaxPool2D(3, padding=3, strides=3)
print(pool(X))
