from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, init, autograd
from mxnet.gluon import nn


class Residual(nn.Block):
    def __init__(self, numchanels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(numchanels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(numchanels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(numchanels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)


net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))


def resnet_block(num_chanels, num_residual, use_first_block=False):
    block = nn.Sequential()
    for i in range(num_residual):
        if i == 0 and not use_first_block:
            block.add(Residual(num_chanels, use_1x1conv=True, strides=2))
        else:
            block.add(Residual(num_chanels))
    return block


net.add(resnet_block(64, 2, use_first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, "output shape:\t", X.shape)
