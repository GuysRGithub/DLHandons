import collections
from d2l import AllDeepLearning as d2l
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
import os
import pandas as pd
import shutil
import time
import math
import tarfile
from pathlib import Path

data_dir = "E:/Python_Data/cifar-10/"
# tiny_data_dir = "E:/Python_Data/kaggle_cifar10_tiny/"
# data_dir = tiny_data_dir
# data_dir = "E:/Python_Data/cifar-10/"
tiny_data_dir = data_dir
a = "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_cifar10_tiny.zip"
demo = False


# def download_voc_pascal(data_dir='../data'):
#     """Download the VOC2012 segmentation dataset."""
#     voc_dir = os.path.join(data_dir, 'Cifar_Tiny')
#     url = "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_cifar10_tiny.zip"
#     sha1 = '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd'
#     fname = gluon.utils.download(url, data_dir, sha1_hash=sha1)
#     with tarfile.open(fname, 'r') as f:
#         f.extractall(data_dir)
#     return voc_dir


def read_csv_labels(fname):
    with open(fname) as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


def copyfile(filename, target_dir):
    Path("%s" % target_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(data_dir + 'train'):
        label = labels[train_file.split('.')[0]]
        fname = data_dir + 'train/' + train_file
        copyfile(fname, data_dir + 'train_valid_test/train_valid/' + label)
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, data_dir + 'train_valid_test/valid/' + label)
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, data_dir + 'train_valid_test/train/' + label)
    return n_valid_per_label


def reorg_test(data_dir):
    for test_file in os.listdir(data_dir + 'test'):
        copyfile(data_dir + 'test/' + test_file,
                 data_dir + 'train_valid_test/test/unknown/')


def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(data_dir + 'trainLabels.csv')
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


batch_size = 1 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(tiny_data_dir, valid_ratio)

transform_train = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(40),
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 0.1), ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                           (0.2023, 0.1994, 0.2010))])
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                           (0.2023, 0.1994, 0.2010))
])

train_ds, valid_ds, train_valid_ds, test_ds = [gluon.data.vision.ImageFolderDataset(
    data_dir+'train_valid_test/'+folder) for folder in ['train', 'valid', 'train_valid', 'test']]

train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='keep') for dataset in (train_ds, train_valid_ds)]

valid_iter, test_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep') for dataset in [valid_ds, test_ds]]


class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1_conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1_conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X, *args, **kwargs):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(),
            nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1_conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net


def get_net(ctx):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(init.Xavier(), ctx=ctx)
    return net


def train(net, train_iter, valid_iter, num_epochs, lr,
          wd, ctx, lr_period, lr_decay):
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr,
                                                          "momentum": 0.9, 'wd': wd})
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for X, y in train_iter:
            y = y.astype('float32').as_in_context(ctx)
            with autograd.record():
                y_hat = net(X.as_in_context(ctx))
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += float(l.sum().asscalar())
            train_acc_sum += float((y_hat.argmax(axis=1) == y).sum().asscalar())
            n += y.size
        time_s = "time %.2f sec" % (time.time() - start)
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter, ctx=ctx)
            epoch_s = ("epoch %d, loss %.2f, train acc %f, valid acc %f, " %
                       (epoch + 1, train_l_sum / n, train_acc_sum / n, valid_acc))
        else:
            epoch_s = ("epoch %d, loss %f, train acc %f" %
                       (epoch + 1, train_l_sum / n, train_acc_sum / n))
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))


ctx, num_epochs, lr, wd = d2l.try_gpu(), 100, 0.1, 5e-4
lr_period, lr_decay, net = 80, 0.1, get_net(ctx)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay)


net, preds = get_net(ctx), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, ctx, lr_period,
      lr_decay)

# for X, _ in test_iter:
#     y_hat = net(X.as_in_context(ctx))
#     preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
# sorted_ids = list(range(1, len(test_ds) + 1))
# sorted_ids.sort(key=lambda x: str(x))
# df = pd.DataFrame({'id': sorted_ids, 'label': preds})
# df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
# df.to_csv('submission.csv', index=False)