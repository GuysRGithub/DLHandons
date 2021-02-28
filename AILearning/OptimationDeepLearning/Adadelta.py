
"""
st = pst-1 + (1-p)gt^2
gt' = sqrt((delta_xt-1 + eps) / ( st + eps)) * gt
xt = xt-1 -gt'
delta_xt = p*delta_xt-1 + (1-p)xt^2

The difference to before is that we perform updates with
the rescaled gradient  g′t  which is computed by taking the
ratio between the average squared rate of change and the average
second moment of the gradient.
"""

from d2l import AllDeepLearning as d2l
from mxnet import nd
from AI.AILearning.OptimationDeepLearning import MinibatchSGD as load


def init_adadelta_states(feature_dim):
    s_w, s_b = nd.zeros((feature_dim, 1)), nd.zeros(1)
    delta_w, delta_b = nd.zeros((feature_dim, 1)), nd.zeros(1)
    return (s_w, delta_w), (s_b, delta_b)


def adadelta(params, states, hyper_params):
    rho, eps = hyper_params['rho'], 1e-6
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1-rho) * nd.square(p.grad)
        g = (nd.sqrt(delta + eps) / nd.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g ** 2


data_iter, feature_dim = load.get_data_ch11(10)
# d2l.train_ch10(adadelta, init_adadelta_states(feature_dim),
#                {'rho': 0.9}, data_iter, feature_dim)
d2l.train_gluon_ch10('adadelta', {'rho': 0.2}, data_iter)
d2l.plt.show()
