from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, autograd, init, lr_scheduler
from mxnet.gluon import nn


net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
ctx = d2l.try_gpu()
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256)


def train(net, train_iter, test_iter, num_epochs, loss, trainer, ctx):
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch',
                            xlim=[0, num_epochs], legend=['train loss', 'train acc', 'test acc'],)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum().asscalar(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
            if (i+1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print('train loss %.3f, train acc %.3f, test acc %.3f' % (train_loss, train_acc, test_acc))


class SquareRootScheduler(object):
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update, *args, **kwargs):
        return self.lr * pow(num_update + 1, -0.5)


lr, num_epochs = 0.5, 40
# net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
# train(net, train_iter, test_iter, num_epochs, loss, trainer, ctx)
# d2l.plt.show()

scheduler = SquareRootScheduler(lr=1.0)
d2l.plot(nd.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
d2l.plt.show()

