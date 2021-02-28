from mxnet import gluon, init, nd
from mxnet.gluon import nn


def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net


net = get_net()
print(net.collect_params())
