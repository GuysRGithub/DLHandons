"""
    RMSProp (fix AdaGrad scale down lr too fast)
        s = Bs + (1-B)*g(J(0))*g(J(0))
        0 = 0 - n*g(J(0)) / sqrt(s+e)

    st←γs_t−1+(1−γ)g^2_t,
    xt←xt−1−η/sqrt(st+ϵ)⊙gt.
"""


from d2l import AllDeepLearning as d2l
import math
from mxnet import nd
from AI.AILearning.OptimationDeepLearning import MinibatchSGD as load

d2l.set_figsize((3.5, 2.5))


def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2 ,s1 ,s2


gamma = 0.5
eta = 0.1


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def init_rmsprop_states(feature_dim):
    s_w = nd.zeros((feature_dim, 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)


def rmsprop(params, states, hyper_params):
    gamma, eps = hyper_params['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1-gamma) * nd.square(p.grad)
        p[:] -= hyper_params["lr"] * p.grad / nd.sqrt(s + eps)


data_iter, feature_dim = load.get_data_ch11(10)
d2l.train_ch10(rmsprop,
               init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 1}, data_iter, feature_dim)


d2l.plt.show()

