from d2l import  AllDeepLearning as d2l
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import nn

d2l.set_figsize((3.5, 2.5))
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
# img = image.imread('E:\\Python_Data\\jisoo.jpg')
# d2l.plt.imshow(img)
d2l.plt.show()


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)


def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
                                 batch_size=batch_size, shuffle=is_train,
                                 num_workers=d2l.get_dataloader_workers())


def train_batch_ch13(net, features, labels, loss, trainer, ctx_list, split_f=d2l.split_batch):
    Xs, ys = split_f(features, labels, ctx_list)
    with autograd.record():
        pys = [net(X) for X in Xs]
        ls = [loss(py, y) for py, y in zip(pys, ys)]
    for l in ls:
        l.backward()
    trainer.step(features.shape[0])
    train_loss_sum = sum([float(l.sum().asscalar()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(py, y) for py, y in zip(pys, ys))
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, ctx_list=d2l.try_all_gpus(),
               split_f=d2l.split_batch):
    num_batchs, timer = len(train_iter), d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    metric = d2l.Accumulator(4)
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss,
                                      trainer, ctx_list, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i+1) % (num_batchs // 5) == 0:
                animator.add(epoch+i/num_batchs,
                             (metric[0]/metric[2], metric[1]/metric[3]))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print('loss %.3f, train acc %.3f, test acc %.3f' % (metric[0]/metric[2], metric[1]/metric[3], test_acc))
    print('%.1f examples/sec on %s' % (metric[2]*num_epochs/timer.sum(), ctx_list))


batch_size, ctx, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=ctx)
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])


def train_with_data_aug(train_augs, test_augs, net, lr=.001):
    train_iter = load_cifar10(True, train_augs, batch_size=batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size=batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, ctx)


train_with_data_aug(train_augs, test_augs, net, lr=.001)
d2l.plt.show()
