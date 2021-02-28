from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet import init
from d2l import AllDeepLearning as d2l

true_w = nd.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a Gluon data loader"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
epoch = 3
for i in nd.arange(1, epoch + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch: %d, loss: %f' % (epoch, l.mean().asnumpy()))
w = net[0].weight.data()
print('Error in estimating true_w:', true_w.reshape(w.shape) - w)
b = net[0].bias.data()
print('Error in estimating bia:', true_b - b)

