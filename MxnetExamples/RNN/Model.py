import mxnet as mx
import mxnet.symbol as S
import numpy as np


def cross_entropy_loss(inputs, labels, rescale_loss=1):
    """Cross entropy loss with mask"""
    criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss(weight=rescale_loss)
    loss = criterion(inputs, labels)
    mask = S.var('mask')
    loss = loss * S.reshape(mask, shape=(-1,))
    return S.make_loss(loss.mean())


def rnn(bptt, vocab_size, num_embed, num_hid, num_layers, dropout, num_proj, batch_size):
    """Word embedding with LSTM Projected"""
    state_names = []
    data = S.var('data')
    weight = S.var("encoder_weight", stype='row_sparse')
    embed = S.sparse.Embedding(data, weight, input_dim=vocab_size,
                               output_dim=num_embed, name='embed',
                               sparse_grad=True)

    states = []
    outputs = S.Dropout(embed, p=dropout)
    for i in range(num_layers):
        prefix = 'lstmp%d' % i
        init_h = S.var(prefix + 'init_h', shape=(batch_size, num_proj),
                       init=mx.init.Zero())
        init_c = S.var(prefix + 'init_c', shape=(batch_size, num_hid),
                       init=mx.init.Zero())
        state_names += [prefix + 'init_h', prefix + 'init_c']
        lstmp = mx.gluon.contrib.rnn.LSTMPCell(num_hid, num_proj,
                                               prefix=prefix)
        outputs, next_state = lstmp.unroll(bptt, outputs, begin_state=[init_h, init_c],
                                           layout='NTC', merge_outputs=True)
        outputs = S.Dropout(outputs, p=dropout)
        states += [S.stop_gradient(s) for s in next_state]
    outputs = S.reshape(outputs, shape=(-1, num_proj))

    trainable_lstm_args = []
    for arg in outputs.list_arguments():
        if 'lstmp' in arg and 'init' not in arg:
            trainable_lstm_args.append(arg)
    return outputs, states, trainable_lstm_args, state_names


def sampled_softmax(num_classes, num_samples, in_dim, inputs, weight, bias,
                    sampled_values, remove_accidental_hits=True):
    sample, prob_sample, prob_target = sampled_values
    # num samples
    sample = S.var('sample', shape=(num_samples,), dtype='float32')
    label = S.var('label')
    label = S.reshape(label, shape=(-1,), name='label_reshape')
    sample_label = S.concat(sample, label, dim=0)
    sample_target_w = S.sparse.Embedding(data=sample_label, weight=weight,
                                         input_dim=num_classes,
                                         output_dim=in_dim,
                                         sparse_grad=True)
    sample_target_b = S.sparse.Embedding(sample_label, weight=bias,
                                         input_dim=num_classes,
                                         output_dim=1,
                                         sparse_grad=True)

    sample_w = S.slice(sample_target_w, begin=(0, 0),
                       end=(num_samples, None))
    target_w = S.slice(sample_target_w, begin=(num_samples, 0),
                       end=(None, None))
    sample_b = S.slice(sample_target_b, begin=(0, 0),
                       end=(num_samples, None))
    target_b = S.slice(sample_target_b, begin=(num_samples, 0),
                       end=(None, None))

    true_pred = S.sum(target_w * inputs, axis=1, keepdims=True) + target_b

    sample_b = S.reshape(sample_b, (-1,))
    sample_pred = S.FullyConnected(inputs, weight=sample_w, bias=sample_b,
                                   num_hidden=num_samples)

    if remove_accidental_hits:
        label_v = S.reshape(label, (-1, 1))
        sampled_v = S.reshape(sample, (1, -1))
        neg = S.broadcast_equal(label_v, sampled_v) * -1e37
        sample_pred = sample_pred + neg

    prob_sample = S.reshape(prob_sample, shape=(1, num_samples))
    p_target = true_pred - S.log(prob_target)
    p_sample = S.broadcast_sub(sample_pred, S.log(prob_sample))

    logits = S.concat(p_target, p_sample, dim=1)
    new_targets = S.zeros_like(label)
    return logits, new_targets


def generate_samples(label, num_splits, sampler):

    def listify(x):
        return x if isinstance(x, list) else [x]

    label_splits = listify(label.split(num_splits, axis=0))
    prob_samples = []
    prob_targets = []
    samples = []
    for label_split in label_splits:
        label_split_2d = label_split.reshape((-1, 1))
        sample_value = sampler.draw(label_split_2d)
        sampled_classes, exp_cnt_true, exp_cnt_sampled = sample_value
        samples.append(sampled_classes.astype(np.float32))
        prob_targets.append(exp_cnt_true.astype(np.float32).reshape((-1, 1)))
        prob_samples.append(exp_cnt_sampled.astype(np.float32))
    return samples, prob_samples, prob_targets


class Model:

    def __init__(self, n_tokens, rescale_loss, bptt, em_size, num_hid, num_layers,
                 dropout, num_proj, batch_size, k):
        out = rnn(bptt, n_tokens, em_size, num_hid, num_layers, dropout,
                  num_proj, batch_size)
        rnn_out, self.last_states, self.lstm_args, self.state_names = out
        decoder_w = S.var("decoder_weight", stype='row_sparse')
        decoder_b = S.var("decoder_bias", shape=(n_tokens, 1), stype="row_sparse")
        sample = S.var('sample', shape=(k,))
        prob_sample = S.var('prob_sample', shape=(k, ))
        prob_target = S.var("prob_target")
        self.sample_names = ['sample', 'prob_sample', 'prob_target']
        logits, new_targets = sampled_softmax(n_tokens, k, num_proj,
                                              rnn_out, decoder_w, decoder_b,
                                              [sample, prob_sample, prob_target])
        self.train_loss = cross_entropy_loss(logits, new_targets, rescale_loss)

        eval_logits = S.FullyConnected(data=rnn_out, weight=decoder_w,
                                       num_hidden=n_tokens, name='decode_fc',
                                       bias=decoder_b)
        label = S.Variable('label')
        label = S.reshape(label, shape=(-1,))
        self.eval_loss = cross_entropy_loss(eval_logits, label)

    def eval(self):
        return S.Group(self.last_states + [self.eval_loss])

    def train(self):
        return S.Group(self.last_states + [self.train_loss])

