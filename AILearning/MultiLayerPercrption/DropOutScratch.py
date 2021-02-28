from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, autograd
from mxnet.gluon import nn


def drop_out(X, prob_drop):
    assert 0 <= prob_drop <= 1
    if prob_drop == 1:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) > prob_drop
    return mask * X / (1.0 - prob_drop)


X = nd.arange(16).reshape((2, 8))
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hidden1))
b1 = nd.zeros(num_hidden1)
W2 = nd.random.normal(scale=0.01, shape=(num_hidden1, num_hidden2))
b2 = nd.zeros(num_hidden2)
W3 = nd.random.normal(scale=0.01, shape=(num_hidden2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

drop_prob1, drop_prob2 = 0.2, 0.5


def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():
        H1 = drop_out(H1, drop_prob1)
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        H2 = drop_out(H2, drop_prob2)
    return nd.dot(H2, W3) + b3


num_epochs, lr, batch_size = 10, 0.05, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, lambda batch_size: d2l.sgd(params, lr, batch_size))
d2l.plt.show()



