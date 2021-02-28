import mxnet
from d2l import AllDeepLearning as d2l
from mxnet import nd, autograd, gluon
import sys

mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)

'''
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


X, y = mnist_train[:18]
d2l.show_images(X.squeeze(axis=-1), 2, 9, get_fashion_mnist_labels(y))
#d2l.plt.show()

batch_size = 256
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=get_dataloader_workers())
'''


def get_dataloader_workers(num_workers=4):
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers


def load_data_fashion_mnist(batch_size, resize=None):
    data_set = gluon.data.vision
    trans = [data_set.transforms.Resize(resize)] if resize else []
    trans.append(data_set.transforms.ToTensor())
    trans = data_set.transforms.Compose(trans)
    mnist_train = data_set.FashionMNIST(train=True).transform_first(trans)
    mnist_test = data_set.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, (64, 64))
for X, y in train_iter:
    print(y)
    break
