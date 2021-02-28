from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd
from mxnet.gluon import nn


def conv_block(num_chanels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_chanels, kernel_size=3, padding=1))
    return blk


class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_chanels, **kwargs ):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_chanels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concatenate([X, Y], axis=1)
        return X


blk = DenseBlock(2, 10)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 8, 8))
Y = blk(X)
print(Y.shape)


def transition_block(num_chanels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_chanels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk


blk = transition_block(10)
blk.initialize()
print(blk(Y).shape)


net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    num_channels = 64
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that haves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))

net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))

# X = nd.random.uniform(shape=(1, 1, 224, 224))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'output shape:\t', X.shape)
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch5(net, train_iter, test_iter, num_epochs, lr)
