from d2l import AllDeepLearning as d2l
from mxnet.gluon import nn, rnn
from mxnet import nd, init, gluon, autograd


class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        print("Encode X", X)
        X = self.embedding(X)

        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.context)
        out, state = self.rnn(X, state)
        return out, state


# encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# encoder.initialize()
# X = nd.zeros((4, 7))
# output, state = encoder(X)
# print(output.shape)


class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size,
                 num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embbeding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embbeding(X).swapaxes(0, 1)
        out, state = self.rnn(X, state)
        out = self.dense(out).swapaxes(0, 1)
        return out, state


# decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8,
#                          num_hiddens=16, num_layers=2)
# # decoder.initialize()
# # state = decoder.init_state(encoder(X))
# # out, state = decoder(X, state)
# # print(out.shape, len(state), state[0].shape, state[1].shape)


class MaskedSoftMaxCELoss(gluon.loss.SoftmaxCELoss):
    def forward(self, pred, label, valid_length):
        weights = nd.expand_dims(nd.ones_like(label), axis=-1)
        weights = nd.SequenceMask(weights, valid_length, True, axis=1)
        return super(MaskedSoftMaxCELoss, self).forward(pred, label, weights)


def train_s2s_ch9(model, data_iter, lr, num_epochs, ctx):
    model.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),
                            'adam', {'learning_rate': lr})
    loss = MaskedSoftMaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel="loss",
                            xlim=[1, num_epochs], ylim=[0, 0.25])
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            X, X_vlen, Y, Y_vlen = [x.as_in_context(ctx) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen - 1
            with autograd.record():
                Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)
                l = loss(Y_hat, Y_label, Y_vlen)
            l.backward()
            d2l.grad_clipping(model, 1)
            num_tokens = Y_vlen.sum().asscalar()
            trainer.step(num_tokens)
            metric.add(l.sum().asscalar(), num_tokens)
        if epoch % 10 == 0:
            animator.add(epoch, (metric[0] / metric[1]))
    print('loss %.3f, %d tokens/sec on %s' % (metric[0] / metric[1], metric[1] / timer.stop(), ctx))


def predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, num_steps, ctx):
    src_tokens = src_vocab[src_sentence.lower().split(' ')]  # use index to train
    enc_valid_length = nd.array([len(src_tokens)], ctx=ctx)
    src_tokens = d2l.trim_pad(src_tokens, num_steps, src_vocab.pad)
    enc_X = nd.array(src_tokens, ctx=ctx)
    enc_outputs = model.encoder(nd.expand_dims(enc_X, axis=0),
                                enc_valid_length)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_length)
    dec_X = nd.expand_dims(nd.array([tgt_vocab.bos], ctx=ctx), axis=0)
    predict_tokens = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        dec_X = Y.argmax(axis=2)
        py = dec_X.squeeze(axis=0).astype('int32').asscalar()
        if py == tgt_vocab.eos:
            break
        predict_tokens.append(py)
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))


embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.0
batch_size, num_steps = 64, 10
lr, num_epochs, ctx = 0.005, 300, d2l.try_gpu()
src_vocab, tgt_vocab, train_iter = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
encoder.initialize()
decoder = Seq2SeqDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout
)
decoder.initialize()
model = d2l.EncoderDecoder(encoder, decoder)
# train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)
# d2l.plt.show()
for sentence in ['Go .', 'Wow !', "I'm OK .", 'I won !']:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", src_vocab[sentence.lower().split(' ')])
    print(sentence + ' => ' + predict_s2s_ch9(
        model, sentence, src_vocab, tgt_vocab, num_steps, ctx))
