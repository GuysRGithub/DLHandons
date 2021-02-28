from d2l import AllDeepLearning as d2l
from mxnet import nd
from mxnet.gluon import rnn
from AI.AILearning.NaturalLanguageProcessing import NaturalLanguageStatistics as loader

batch_size, num_step = 32, 35
train_iter, vocab = loader.load_data_time_machine(batch_size, num_step)


def get_params(vocab_size, num_hiddens, ctx):
    """

    :type num_hiddens: int
    :type vocab_size: int
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return nd.random.normal(shape=shape, scale=0.01, ctx=ctx)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                nd.zeros(num_hiddens, ctx))

    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = nd.zeros(shape=num_outputs, ctx=ctx)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]

    for param in params:
        param.attach_grad()
    return params


def init_lstm_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),
            nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo,
     W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * nd.tanh(C)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return nd.concat(*outputs, dim=0), (H, C)  # output and new state


vocab_size, num_hiddens, ctx = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(vocab_size, num_hiddens, ctx, get_params, init_lstm_state, lstm)
lstm_layer = rnn.LSTM(num_hiddens, 2)
lstm_model = d2l.RNNModel(lstm_layer, vocab_size)
d2l.train_ch8(lstm_model, train_iter, vocab, lr, num_epochs, ctx)
d2l.plt.show()
