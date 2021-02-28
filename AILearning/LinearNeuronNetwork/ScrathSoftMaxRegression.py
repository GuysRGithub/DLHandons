from mxnet import nd, gluon, autograd
from d2l import AllDeepLearning as d2l
from IPython import display
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)


W.attach_grad()
b.attach_grad()


def soft_max(x):
    x_exp = x.exp()
    partition = x_exp.sum(axis=1, keepdims=True)
    return x_exp / partition


X = nd.random.normal(shape=(2, 5))
X_prob = soft_max(X)
X_prob, X_prob.sum(axis=1)


def net(X):
    return soft_max(nd.dot(X.reshape((-1, num_inputs)), W) + b)


X = nd.random.normal(shape=(2, 5))
X_prob = soft_max(X)
X_prob, X_prob.sum(axis=1)


def cross_entropy(y_hat, y):
    return - nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()


def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    for X, y in data_iter:
        y = y.astype('float32')
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]
    # num_corrected_examples, num_examples


class Accumulator(object):
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0] * len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(l.sum().asscalar(), accuracy(y_hat, y), y.size)
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator(object):
    def __init__(self, x_label=None, y_label=None, legend=[], x_lim=None, y_lim=None, x_scale='linear', y_scale='linear'
                 , fmts=None, n_rows=1, n_cols=1, fig_size=(3.5, 2.5)):
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(n_rows, n_cols, figsize=fig_size)
        if n_rows * n_cols == 1:
            self.axes = [self.axes, ]
        self.config_axis = lambda: d2l.set_axes(self.axes[0], x_label, y_label, x_lim, y_lim, x_scale, y_scale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        if not self.fmts:
            self.fmts = '-' * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axis()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epoch, updater):
    trains, test_acc = [], []
    animator = Animator(x_label='epoch', x_lim=[1, num_epoch], y_lim=[0.3, 0.9]
                        , legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epoch):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc, ))


num_epochs, lr = 10, 0.1
updater = lambda batch_size: d2l.sgd([W, b], lr, batch_size)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
d2l.plt.show()
