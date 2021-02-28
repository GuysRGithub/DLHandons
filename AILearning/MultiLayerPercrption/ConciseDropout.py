from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import nn

drop_prob1, drop_prob2 = 0.2, 0.5
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dropout(drop_prob2)
        , nn.Dense(256, activation='relu'), nn.Dropout(drop_prob2)
        , nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
train_iter, test_iter = d2l.load_data_fashion_mnist(256)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
d2l.train_ch3(net, train_iter, test_iter, loss, 10, trainer)
d2l.plt.show()
