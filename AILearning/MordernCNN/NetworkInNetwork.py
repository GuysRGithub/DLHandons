from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd
from mxnet.gluon import nn


def NIN_block(num_chanels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_chanels, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu'),
            nn.Conv2D(num_chanels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_chanels, kernel_size=1, activation='relu'))
    return blk


net = nn.Sequential()
net.add(NIN_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        NIN_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        NIN_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        NIN_block(10, kernel_size=3, strides=1, padding=1),
        nn.GlobalAvgPool2D(),
        nn.Flatten())
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'Shape:\t', X.shape)

