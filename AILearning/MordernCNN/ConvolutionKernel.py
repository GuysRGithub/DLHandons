from mxnet import autograd, nd
from mxnet.gluon import nn


def comp_conv2d(conv2d, X):
    conv2d.initialize()
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
X = nd.random.uniform(shape=(8, 8))
print(comp_conv2d(conv2d, X))
