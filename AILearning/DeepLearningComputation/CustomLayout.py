from mxnet.gluon import nn
from mxnet import nd, init, gluon


class CustomLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = CustomLayer()
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), CustomLayer())
net.initialize()

params = gluon.ParameterDict()
# noinspection PyTypeChecker
params.get('param2', shape=(2, 3))


class MyDense(nn.Block):
    def __init__(self, in_units, units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get("bias", shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data() + self.bias.data())
        return nd.relu(linear)


dense = MyDense(in_units=3, units=6)
dense.initialize()
net = nn.Sequential()
net.add(nn.Dense(8, in_units=64))
net.add(nn.Dense(2, in_units=8))
net.initialize()
print(net(nd.random.uniform(shape=(2, 64))))
