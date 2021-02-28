from d2l import AllDeepLearning as d2l
from mxnet import nd
from mxnet.gluon import nn


def get_net():
    net = nn.HybridSequential()
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net


class bechmark:
    def __init__(self, description="Done in %.4f sec"):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(self.description % self.timer.stop())


class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x, *args, **kwargs):
        print('module F: ', F)
        print('value x: ', x)
        x = F.relu(self.hidden(x))
        print('result   :', x)
        return self.output(x)


net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 3))
net.hybridize()
print(net(x))

# net = get_net()
# net.hybridize()
# print(net(x))

# 5.2946981e-04  4.1249798e-05
# x = nd.random.normal(shape=(1, 512))
# net = get_net()
# with bechmark('Without hybridize %.4f sec'):
#     for i in range(1000): net(x)
#     nd.waitall()
#
# net.hybridize()
# with bechmark("With hybridize %.4f sec"):
#     for i in range(1000): net(x)
#     nd.waitall()