from mxnet import autograd, gluon, nd, init
from d2l import AllDeepLearning as d2l
from mxnet.gluon import nn
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init=init.Normal(sigma=0.01))
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epochs = 20
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
