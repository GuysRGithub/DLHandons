import mxnet
import matplotlib
from mxnet import nd, gluon, autograd
import math
import random

from AI import AILearning
from d2l import AllDeepLearning as d2l


def synthetic_data(w, b, num_examples):
    """generate y = X w + b + noise"""
    X = nd.random.normal(scale=1, shape=(num_examples, len(w)))
    y = nd.dot(X, w) + b
    y += nd.random.normal(scale=0.01, shape=y.shape)
    return X, y


true_w = nd.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
d2l.set_figsize((3.5, 2.5))
d2l.plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)
        # The “take” function will then return the corresponding element
        # based on the indices


batch_size = 10


for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
w = nd.random.normal(scale=0.01, shape=(2, 1))
b = nd.zeros(shape=(1, ))
w.attach_grad()
b.attach_grad()


def linear_reg(X, w, b):
    return nd.dot(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


lr = 0.03  # Learning rate
num_epochs = 3  # Number of iterations
net = linear_reg  # Our fancy linear model
loss = squared_loss  # 0.5 (y-y')^2

for epoch in range(num_epochs):
    # Assuming the number of examples can be divided by the batch size, all
    # the examples in the training data set are used once in one epoch
    # iteration. The features and tags of mini-batch examples are given by X
    # and y respectively
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in X and y
        l.backward()  # Compute gradient on l with respect to [w,b]
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))


print('Error in estimating w', true_w - w.reshape(true_w.shape))
print('Error in estimating b', true_b - b)