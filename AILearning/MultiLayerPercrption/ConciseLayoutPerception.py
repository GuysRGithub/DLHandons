from mxnet import autograd, nd, init, gluon
from d2l import AllDeepLearning as d2l
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(512, activation='relu'), nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

batch_size, num_epoch = 256, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, trainer)
d2l.plt.show()

