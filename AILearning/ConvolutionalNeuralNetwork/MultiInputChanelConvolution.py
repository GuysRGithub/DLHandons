from mxnet import nd
from d2l import AllDeepLearning as d2l


def multi_input_conv(X, K):
    return nd.add_n(*[d2l.corr2d(x, y) for x, y in zip(X, K)])


def corr2d_multi_in_out(X, K):
    return nd.stack(*[multi_input_conv(X, y) for y in K])


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)
    return Y.reshape((c_o, h, w))


X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().asscalar())


