from mxnet import gluon, init, nd
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

x = nd.random.normal(shape=(2, 20))
net(x)


def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add(block1())
    return net


reg_net = nn.Sequential()
reg_net.add(block2())
reg_net.add(nn.Dense(10))
reg_net.initialize()
reg_net(x)


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print("Init", name, data.shape)
        data[:] = init.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() > 5


net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()
net(x)
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 12
print(net[1].weight.data()[0, 0], net[2].weight.data()[0, 0])
