from mxnet.gluon import nn
import math
from mxnet import nd
from d2l import AllDeepLearning as d2l


def masked_softmax(X, valid_length):
    if valid_length is None:
        return nd.softmax(X)
    else:
        shape = X.shape
        if valid_length.ndim == 1:
            valid_length = valid_length.repeat(shape[1], axis=0)
        else:
            valid_length = valid_length.reshape((-1,))

        X = nd.SequenceMask(X.reshape((-1, shape[-1])), valid_length, True, axis=1, value=-1e6)
        return X.softmax().reshape(shape)


class DotProductAttention(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        scores = nd.batch_dot(query, key, transpose_b=True) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        print(attention_weights)
        return nd.batch_dot(attention_weights, value)


class MLPAttention(nn.Block):
    def __init__(self, units, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.W_k = nn.Dense(units=units, activation='tanh',
                            use_bias=False, flatten=False)
        self.W_q = nn.Dense(units, activation='tanh',
                            use_bias=False, flatten=False)
        self.v = nn.Dense(1, use_bias=False, flatten=False)

    def forward(self, query, key, value, valid_length):
        query, key = self.W_k(query), self.W_q(key)
        print(key)
        features = nd.expand_dims(query, axis=2) + nd.expand_dims(key, axis=1)
        scores = nd.squeeze(self.v(features), axis=-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return nd.batch_dot(attention_weights, value)


attention = MLPAttention(units=8, dropout=0.1)
attention.initialize()
keys = nd.ones((2, 10, 2))
values = nd.arange(40).reshape(1, 10, 4).repeat(2, axis=0)
print(attention(nd.ones((2, 1, 2)), keys, values, nd.array([2, 6])))