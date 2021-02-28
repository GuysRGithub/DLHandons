from mxnet import nd
from d2l import AllDeepLearning as d2l
from mxnet.gluon import nn


def vgg_block(num_convs, num_chanel):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_chanel, kernel_size=3, padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    net = nn.Sequential()
    for (num_convs, num_chanels) in conv_arch:
        net.add(vgg_block(num_convs, num_chanels))

    net.add(nn.Dense(64, activation='relu'), nn.Dropout(0.5),
            nn.Dense(64, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net


net = vgg(conv_arch)
net.initialize()
X = nd.random.uniform(shape=(1, 1, 224, 224))


ration = 64
small_conv_arch = [(pair[0], pair[1] // ration) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.01, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch5(net, train_iter, test_iter, num_epochs, lr)
d2l.plt.show()
print("Complete!")
