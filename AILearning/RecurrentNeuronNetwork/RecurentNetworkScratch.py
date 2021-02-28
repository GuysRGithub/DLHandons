from d2l import AllDeepLearning as d2l
import math
from mxnet import autograd, gluon, init
from mxnet import nd
import AI.AILearning.NaturalLanguageProcessing.NaturalLanguageStatistics as AI
batch_size, num_steps = 32, 35
train_iter, vocab = AI.load_data_time_machine(batch_size, num_steps)
X = nd.arange(batch_size*num_steps).reshape(batch_size, num_steps)


def get_params(vocab_size, num_hiddens, ctx):
    num_inputs = num_outputs = vocab_size
    normal = lambda shape: nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def init_rnn_state(batch_size, num_hiddens, ctx):
    return nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),


def rnn(inputs, state, params):
    # inputs shape: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return nd.concat(*outputs, dim=0), (H,)


class RNNModelScratch(object):
    def __init__(self, vocab_size, num_hidden, ctx, get_params, init_state, forward):
        self.vocab_size, self.num_hidden = vocab_size, num_hidden
        self.params = get_params(vocab_size, num_hidden, ctx=ctx)
        self.init_state, self.forward_fn = init_state, forward

    def __call__(self, X, state):
        X = nd.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hidden, ctx)


vocab_size, num_hidden, ctx = len(vocab), 512, d2l.try_gpu()
model = RNNModelScratch(len(vocab), num_hidden, ctx, get_params, init_rnn_state, rnn)
state = model.begin_state(X.shape[0], ctx)
Y, new_state = model(X.as_in_context(ctx), state)


def predict_ch8(prefix, num_predicts, model, vocab, ctx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: nd.array([outputs[-1]], ctx=ctx).reshape(1, 1)
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_predicts):
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(axis=1).reshape(1).asscalar()))  # error
    return ''.join(vocab.idx_to_token[i] for i in outputs)


def grad_clipping(model, theta):
    if isinstance(model, gluon.Block):
        params = [p.data() for p in model.collect_params().value()]
    else:
        params = model.params
    norm = math.sqrt(sum((p.grad ** 2).sum().asscalar() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(model, train_iter, loss, updater, ctx, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = model.begin_state(batch_size=X.shape[0], ctx=ctx)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():
            py, state = model(X, state)
            l = loss(py, y).mean()
        l.backward()
        grad_clipping(model, 1)
        updater(batch_size=1)
        metric.add(l.asscalar() * y.size, y.size)
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(model, train_iter, vocab, lr, num_epochs, ctx, use_random_iter=False):
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[1, num_epochs])
    if isinstance(model, gluon.Block):
        model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
        trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, model, vocab, ctx)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(model, train_iter, loss, updater, ctx, use_random_iter)

        if epoch % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])

    print("Perplexity %.1f, %d tokens/sec on %s" % (ppl, speed, ctx))
    print(predict('time traveller'))
    print(predict('traveller'))


num_hidden = 512
ctx = d2l.try_gpu()
model = RNNModelScratch(len(vocab), num_hidden, ctx, get_params, init_rnn_state, rnn)
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
d2l.plt.show()
