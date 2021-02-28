from d2l import AllDeepLearning as d2l
from mxnet import nd
from AI.AILearning.NaturalLanguageProcessing import NaturalLanguageStatistics as loader

batch_size, num_step = 32, 25
train_iter, vocab = loader.load_data_time_machine(batch_size, num_step)


def get_params(vocab_size, num_hiddens, ctx):
    num_inputs = num_outputs = vocab_size
    normal = lambda shape: nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    three = lambda: (normal((num_inputs, num_hiddens)),
                     normal((num_hiddens, num_hiddens)),
                     nd.zeros(num_hiddens, ctx=ctx))
    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def init_gru_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)
        H_tilda = nd.tanh(nd.dot(X, W_xh) + nd.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return nd.concat(*outputs, dim=0), (H,)  # not use nd.concatenate and unpack list when pass argument


vocab_size, num_hiddens, ctx = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, ctx, get_params, init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
d2l.plt.show()

