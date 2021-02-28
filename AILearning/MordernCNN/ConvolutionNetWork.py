from mxnet import gluon, nd, autograd
from mxnet.gluon import nn


def corr2D(X, K):
    h, w = K.shape
    Y = nd.zeros(shape=(X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weights = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shpae=(1, ))

    def forward(self, x):
        return corr2D(x, self.weights.data()) + self.bias.data()


X = nd.ones((6, 8))
X[:, 2:6] = 0
K = nd.array([[-1, 1]])
Y = corr2D(X, K)
conv2D = nn.Conv2D(1, (1, 2))
conv2D.initialize()


X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)

for i in range(10):
    with autograd.record():
        Y_hat = conv2D(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    conv2D.weight.data()[:] -= 3e-2 * conv2D.weight.grad()
    if (i + 1) % 2:
        print('batch %d, loss %f' % (i + 1, l.sum().asscalar()))

print(conv2D.weight.data())


