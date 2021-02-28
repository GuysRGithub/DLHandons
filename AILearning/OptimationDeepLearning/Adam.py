"""
    Adam: keep track of exponentially decaying average of past gradients
    and keeps track of exponentially decaying average of past squared gradients
        vt = B_1*v_t-1 + (1 - B1)gt                 }(t: index)
        st = B_2*s_t-1 + (1 - B2)gt^2               }
        v`t = v_t / (1 - B1^t)                      ( ` != ' )
        s`t = s_t / (1 - B2^t)
        g'_t = n*v`t / (sqrt(s`t) + e)
        xt = xt-1 - g'_t
    Which T represents the iteration number
"""

from d2l import AllDeepLearning as d2l
from AI.AILearning.OptimationDeepLearning import MinibatchSGD as load
from mxnet import nd


def init_adam_states(features_dim):
    s_w = nd.zeros((features_dim, 1))
    s_b = nd.zeros(1)
    v_w = nd.zeros((features_dim, 1))
    v_b = nd.zeros(1)
    return (v_w, s_w), (v_b, s_b)


def adam(params, states, hyper_params):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * nd.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyper_params['t'])
        s_bias_corr = s / (1 - beta2 ** hyper_params['t'])
        p[:] -= hyper_params['lr'] * \
            v_bias_corr / (nd.sqrt(s_bias_corr) + eps)

    hyper_params['t'] += 1


data_iter, feature_dim = load.get_data_ch11(batch_size=10)
# d2l.train_ch10(adam, init_adam_states(feature_dim),
#                {'t': 1, "lr": 0.01}, data_iter, feature_dim)
d2l.train_gluon_ch10('adam', {"learning_rate": 0.01}, data_iter)
d2l.plt.show()

