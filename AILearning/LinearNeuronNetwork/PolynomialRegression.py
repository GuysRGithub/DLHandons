from mxnet import nd, gluon
import mxnet
from d2l import AllDeepLearning as d2l
from mxnet.gluon import nn

max_degree = 20
n_train, n_test = 100, 100
true_w = nd.zeros(max_degree)
true_w[0:4] = nd.array([5, 1.2, -3.4, 5.6])

features = nd.random.normal(shape=(n_train + n_test, 1))
features = nd.random.shuffle(features)
poly_features = nd.power(features, nd.arange(max_degree).reshape(1, -1))
poly_features = poly_features / (nd.gamma((nd.arange(max_degree) + 1).reshape(1, -1)))
labels = nd.dot(poly_features, true_w)
labels += nd.random.normal(scale=0.01, shape=labels.shape)
x = nd.arange(20).reshape(4, 5)


def evaluate_loss(net, test_data, loss):
    metric = d2l.Accumulator(2)
    for X, y in test_data:
        metric.add(loss(net(X), y).sum().asscalar(), y.size)
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels, num_epoch=1000):
    net = nn.Sequential()
    loss = gluon.loss.L2Loss()

    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size, is_train=True)
    test_iter = d2l.load_array((test_features, test_labels), batch_size, is_train=False)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=[1, num_epoch], ylim=[1e-3, 1e-5]
                            , legend=['train', 'test'])

    for epoch in range(1, num_epoch + 1):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch % 50 == 0:
            animator.add(epoch, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
    print('weight', net[0].weight.data().asnumpy())


n_subnet, n_degree = 120, 4
train(poly_features[:n_subnet, 0:n_degree], poly_features[n_subnet:, 0:n_degree], labels[:n_subnet], labels[n_subnet:])
d2l.plt.show()

