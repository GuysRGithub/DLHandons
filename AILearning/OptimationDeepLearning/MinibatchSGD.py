from d2l import AllDeepLearning as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
import numpy as np

"""

Mini batches
    w←w−η_t*g_t
    where gt=∂_wf(x_t,w)
    
We can increase the computational efficiency of this operation by 
applying it to a minibatch of observations at a time. That is, we 
replace the gradient gt over a single observation by one over a small batch
    gt= 1/|B_t| * ∂w∑i∈B_t(f(xi,w))


"""


def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt("E:\\Python_Data\\airfoil_self_noise.dat",
                         dtype=np.float32)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1] - 1


def sgd(params, states, hyper_params):
    for p in params:
        p[:] -= hyper_params['lr'] * p.grad


def train_ch11(trainer_fn, state, hyper_params, data_iter, feature_dim, num_epochs=2):
    w = nd.random.normal(scale=0.01, shape=(feature_dim, 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs],
                            ylim=[0.22, 0.32])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], state, hyper_params)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter),
                             d2l.evaluate_loss(net, data_iter, loss))
                timer.start()
    print('loss: %.3f, %.3f sec/epoch' % (animator.Y[0][-1], timer.avg()))
    return timer.cumsum(), animator.Y[0]


def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)


gd_res = train_sgd(1, 1500, 10)
sgd_res = train_sgd(0.005, 1)
mini1_res = train_sgd(.4, 100)
mini2_res = train_sgd(.05, 10)
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size = 100', 'batch size = 10'])
d2l.plt.gca().set_xscale('log')
d2l.plt.show()


def train_gluon_ch11(tr_name, hyper_params, data_iter, num_epochs=2):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyper_params)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs],
                            ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss)))
                timer.start()
    print('loss: %.3f, %.3f sec/epoch' % (animator.Y[0][-1], timer.avg()))


data_iter, _ = get_data_ch11(10)
train_gluon_ch11('sgd', {'learning_rate': 0.05}, data_iter)
d2l.plt.show()


"""

    Vectorization makes code more efficient due to reduced overhead 
    arising from the deep learning framework and due to better memory 
    locality and caching on CPUs and GPUs.
    
    There is a trade-off between statistical efficiency arising 
    from SGD and computational efficiency arising from processing large 
    batches of data at a time.
    
    Minibatch stochastic gradient descent offers the best of 
    both worlds: computational and statistical efficiency.
    
    In minibatch SGD we process batches of data obtained by a random 
    permutation of the training data (i.e., each observation is processed only 
    once per epoch, albeit in random order).
    
    It is advisable to decay the learning rates during training.
    
    In general, minibatch SGD is faster than SGD and gradient descent 
    for convergence to a smaller risk, when measured in terms of clock time.

"""