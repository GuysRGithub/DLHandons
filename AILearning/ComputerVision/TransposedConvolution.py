# See img for illustrate

from mxnet import nd, init
from mxnet.gluon import nn
from d2l import AllDeepLearning as d2l


def trans_conv(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y


def kernel2_matrix(K):
    k, W = nd.zeros(5), nd.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W


X = nd.arange(150).reshape(1, 1, 10, 15)
f = nd.random.normal(shape=(1, 1, 64, 64))
# K = nd.array([[0, 1], [2, 3]])
# Y = d2l.corr2d(X, K)
# W = kernel2_matrix(K)
# X = K

t = nn.Conv2DTranspose(1, kernel_size=64, strides=32, padding=16)
t.initialize(init.Constant(f))
print(t(X).shape)
# print(nd.dot(W.T, X.reshape(-1)).reshape(3, 3))