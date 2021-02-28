from mxnet import nd, gluon
from mxnet.gluon import nn
from d2l import AllDeepLearning as d2l

net = nn.Sequential()
net.add(nn.Conv2D(96, 11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=(3, 3), strides=2),
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(10))


X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch5(net, train_iter, test_iter, num_epochs, lr)
d2l.plt.show()
print("Train")
