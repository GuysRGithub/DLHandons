from mxnet import nd, gluon, autograd
from d2l import AllDeepLearning as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hidden = 784, 10, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hidden))
b1 = nd.zeros(num_hidden)
W2 = nd.random.normal(scale=0.01, shape=(num_hidden, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


def relu(x):
    return nd.maximum(0, x)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2


loss = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 10, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, lambda batch_size: d2l.sgd(params, lr, batch_size))
d2l.predict_ch3(net, test_iter)
d2l.plt.show()
