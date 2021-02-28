import math

from d2l import AllDeepLearning as d2l
from mxnet import nd, autograd
from mxnet.gluon import nn


# project keys, values, query, concatenate and dense output
# The self-attention model is a normal attention model, with its query, its key,
# and its value being copied exactly the same from each item of the sequential inputs
class MultiHeadAttention(nn.Block):
    def __init__(self, hidden_size, num_heads, dropout, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(hidden_size, use_bias=False, flatten=False)
        self.W_k = nn.Dense(hidden_size, use_bias=False, flatten=False)
        self.W_v = nn.Dense(hidden_size, use_bias=False, flatten=False)
        self.W_o = nn.Dense(hidden_size, use_bias=False, flatten=False)

    def forward(self, query, key, value, valid_length, *args):
        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)
        if valid_length is not None:
            if valid_length.ndim == 1:
                valid_length = nd.tile(valid_length, self.num_heads)
            else:
                valid_length = nd.tile(valid_length, (self.num_heads, 1))
        output = self.attention(query, key, value, valid_length)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# transpose shape key value query
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = nd.transpose(X, axes=(0, 2, 1, 3))
    output = X.reshape(-1, X.shape[2], X.shape[3])
    return output


# reverse transpose qkv ( transpose output )
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = nd.transpose(X, axes=(0, 2, 1, 3))
    return X.reshape(X.shape[0], X.shape[1], -1)


# 2 dense wise point ( apply two dense to final dimension )
class PositionWiseFFN(nn.Block):
    def __init__(self, ffn_hidden_size, hidden_size_out, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.ffn_1 = nn.Dense(ffn_hidden_size, flatten=False, activation='relu')
        self.ffn_2 = nn.Dense(hidden_size_out, flatten=False)

    def forward(self, X):
        return self.ffn_2(self.ffn_1(X))


# normalize layer ( connect layers smoothly )
class AddNorm(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm()

    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)


# P + X ( P is position value )
class PositionalEncoding(nn.Block):
    def __init__(self, embedding_size, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = nd.zeros((1, max_len, embedding_size))
        X = nd.arange(0, max_len).reshape(-1, 1) / nd.power(
            10000, nd.arange(0, embedding_size, 2)/embedding_size)
        self.P[:, :, 0::2] = nd.sin(X)
        self.P[:, :, 1::2] = nd.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_context(X.context)
        return self.dropout(X)


# This encoder contains a multi-head attention layer, a position-wise feed-forward network,
# and two “add and norm” connection blocks.
class EncoderBlock(nn.Block):
    def __init__(self, embedding_size, ffn_hidden_size, num_heads, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(embedding_size, num_heads, dropout)  # dense, transpose, and concat
        self.add_norm_1 = AddNorm(dropout)  # layer norm, prevent change too much
        self.ffn = PositionWiseFFN(ffn_hidden_size, embedding_size)  # 2 dense
        self.add_norm_2 = AddNorm(dropout)

    def forward(self, X, valid_length):
        Y = self.add_norm_1(X, self.attention(X, X, X, valid_length))
        return self.add_norm_2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embedding_size, ffn_hidden_size,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(EncoderBlock(
                embedding_size, ffn_hidden_size, num_heads, dropout))

    def forward(self, X, valid_length, *args):
        X = self.pos_encoding(self.embed(X) * math.sqrt(self.embedding_size))
        for blk in self.blks:
            X = blk(X, valid_length)
        return X


class DecoderBlock(nn.Block):
    def __init__(self, embedding_size, ffn_hidden_size, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention_1 = MultiHeadAttention(embedding_size, num_heads, dropout)
        self.add_norm_1 = AddNorm(dropout)
        self.attention_2 = MultiHeadAttention(embedding_size, num_heads, dropout)
        self.add_norm_2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_hidden_size, embedding_size)
        self.add_norm_3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_length = state[0], state[1]
        # state[2][i] contains the past queries for this block
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = nd.concat(*(state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if autograd.is_training():
            batch_size, seq_len, _ = X.shape
            valid_length = nd.tile(nd.arange(1, seq_len+1, ctx=X.context), (batch_size, 1))
        else:
            valid_length = None
        X2 = self.attention_1(X, key_values, key_values, valid_length)
        Y = self.add_norm_1(X, X2)
        Y2 = self.attention_2(Y, enc_outputs, enc_outputs, enc_valid_length)
        Z = self.add_norm_2(Y, Y2)
        return self.add_norm_3(Z, self.ffn(Z)), state


class TransformerDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embedding_size,
                 ffn_hidden_size, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(DecoderBlock(
                embedding_size, ffn_hidden_size, num_heads, dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_length, *args):
        return [enc_outputs, enc_valid_length, [None]*self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embed(X) * math.sqrt(self.embedding_size))
        for blk in self.blks:
            X, state = blk(X, state)
        return self.dense(X), state


embed_size, embedding_size, num_layers, dropout = 32, 32, 2, 0
batch_size, num_steps = 64, 10
lr, num_epochs, ctx = 0.005, 100, d2l.try_gpu()
num_hiddens, num_heads = 64, 4
src_vocab, tgt_vocab, train_iter = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(len(src_vocab), embedding_size,
                             num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(src_vocab), embedding_size, num_hiddens,
                             num_heads, num_layers, dropout)
model = d2l.EncoderDecoder(encoder, decoder)
d2l.train_s2s_ch8(model, train_iter, lr, num_epochs, ctx)
for sentence in ['Move .', 'Go !', "I'm fine .", 'I fail !', 'I won', 'I am happy']:
    print(sentence + ' => ' + d2l.predict_s2s_ch8(
        model, sentence, src_vocab, tgt_vocab, num_steps, ctx))
# d2l.plt.show()
# cell = MultiHeadAttention(100, 10, 0.5)
# cell.initialize()
# X = nd.ones((2, 4, 5))
# valid_length = nd.array([2, 3])
# ffn = PositionWiseFFN(4, 8)
# ffn.initialize()
# layer = nn.LayerNorm()
# layer.initialize()
# batch = nn.BatchNorm()
# batch.initialize()
# add_norm = AddNorm(0)
# add_norm.initialize()
# T = nd.array([[1, 2], [2, 4]])
#
# pe = PositionalEncoding(20, 0)
# pe.initialize()
# Y = pe(nd.zeros((1, 100, 20)))

# valid_length = nd.array([2, 100])
#
# X = nd.ones((2, 100, 24))


# encoder_blk = EncoderBlock(24, 48, 8, 0.5)
# encoder_blk.initialize()
# # print(encoder_blk(X, valid_length))
#
#
# valid_length = nd.array([2, 3])
# # #
# # # encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
# # # encoder.initialize()
# # # print(encoder(nd.ones((2, 100)), valid_length))
#
# decoder_blk = DecoderBlock(24, 48, 8, 0.5, 0)
# decoder_blk.initialize()
# X = nd.ones((2, 100, 24))
# state = [encoder_blk(X, valid_length), valid_length, [None]]
# print(decoder_blk(X, state)[0].shape)