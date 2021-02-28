from d2l import AllDeepLearning as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn

T = 1000
time = nd.arange(0, T)
x = nd.sin(0.01 * time) + 0.2 * nd.random.normal(shape=T)
tau = 4
features = nd.zeros((T-tau, tau))
for i in range(tau):
    features[:, i] = x[i: T-tau+i]
labels = x[tau:]
batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
test_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=False)


def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(10, activation='relu'),
            nn.Dense(1))
    net.initialize(init.Xavier())
    return net


loss = gluon.loss.L2Loss()


def train_net(net, train_iter, test_iter, loss, epochs, lr):
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    for epoch in range(1, epochs+1):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        print('epoch: %d, loss: %f' % (epoch, d2l.evaluate_loss(net, train_iter, loss)))


net = get_net()
prediction = nd.zeros(T)
prediction[:n_train] = x[:n_train]
for i in range(n_train, T):
    prediction[i] = net(prediction[i-tau:i].reshape(1, -1)).reshape(1)

train_net(net, train_iter, test_iter, loss, 10, 0.01)
estimates = net(features)
k = 33
features = nd.zeros((k, T-k))
for i in range(tau):
    features[i] = x[i:T-k+i]
for i in range(tau, k):
    features[i] = net(features[i-tau:i].T).T

steps = [4, 8, 16, 32]
d2l.plot([time[i:T-k+i] for i in steps], [features[i] for i in steps],
         legend=["step %d" % i for i in steps], figsize=(4.5, 2.5))
d2l.plt.show()
#
# d2l.plot([time, time[tau:], time[n_train:]],
#          [x, estimates, prediction[n_train:]],
#          legend=["data", "estimates", "multistep"])
# d2l.plt.show()
