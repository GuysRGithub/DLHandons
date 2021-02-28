"""
Momentum:
    At each iteration, it adds the local gradient to the momentum vector m, and update
    its weights by subtracting this momentum vector
        m = Bm + n*g(J(0)) (B: momentum, n: lr)
        0 = 0 - m

Nesterov Accelerated (Momentum variant)
    Measure the gradient of the cost function ahead in the direction of momentum
        m = Bm + n*g(J(0 + Bm)) (B: momentum, n: lr)
        0 = 0 - m

Leaky Averages

gt  =∂w(1/Bt)∑i∈B_t(f(xi,wt−1))
    =1|Bt|∑i∈B_t(g_(i,t−1)).


vt=βv_(t−1) + g_(t,t−1)

vt  =β2v_(t−2) + βg_(t−1,t−2) + g_(t,t−1)=…=
    ∑τ=0->t−1(β^τg_(t−τ,t−τ−1)).
"""

from d2l import AllDeepLearning as d2l
from mxnet import nd
from AI.AILearning.OptimationDeepLearning import MinibatchSGD as load


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0


def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2


eta, beta = 0.6, 0.5
# # d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
# gammas = [0.95, 0.9, 0.6, 0]
# d2l.set_figsize((3.5, 2.5))
# for gamma in gammas:
#     x = nd.arange(40).asnumpy()
#     d2l.plt.plot(x, gamma ** x, label='gamma = %.2f' % gamma)
# d2l.plt.xlabel('time')
# d2l.plt.legend();
# d2l.plt.show()

data_iter, feature_dim = load.get_data_ch11(batch_size=10)


def init_momentum_states(feature_dim):
    v_w = nd.zeros((feature_dim, 1))
    v_b = nd.zeros(1)
    return v_w, v_b


def sgd_momentum(params, states, hyper_params):
    for p, v in zip(params, states):
        v[:] = hyper_params['momentum'] * v + p.grad
        p[:] -= hyper_params['lr'] * v


def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch10(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter, feature_dim, num_epochs)


train_momentum(0.01, 0.9)
d2l.train_gluon_ch10()
