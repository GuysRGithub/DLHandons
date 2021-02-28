"""
    AdaGrad
        Scaling down the gradient vector
            s = s + g(J(0)) * g(J(0)) (square each element in g(J(0))
            0 = 0 - n*g(J(0)) / sqrt(s+e) (divide each element by sqrt(s+e))
"""

import math
from d2l import AllDeepLearning as d2l
from mxnet import nd
from AI.AILearning.OptimationDeepLearning import MinibatchSGD as load


def adagrad_2d(x1, x2, s1, s2):
    """
    gt=∂wl(yt,f(xt,w)),
    st=st−1+gt^2,
    wt=wt−1−η/sqrt(st+ϵ)⋅gt.
    """
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


eta = 0.4


def init_adagrad_states(feature_dim):
    s_w = nd.zeros((feature_dim, 1))
    s_b = nd.zeros(1)
    return s_w, s_b


def adagrad(params, states, hyper_params):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += nd.square(p.grad)
        p[:] -= hyper_params['lr'] * p.grad / nd.sqrt(s + eps)


data_iter, feature_dim = load.get_data_ch11(batch_size=10)
d2l.train_gluon_ch10("adagrad", {'learning_rate': 0.1}, data_iter)

d2l.plt.show()
d2l.gluon.Trainer