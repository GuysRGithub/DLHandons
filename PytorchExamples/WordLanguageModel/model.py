import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, n_token, n_inp, n_hid, n_layers, dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_token, n_inp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(n_inp, n_hid, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option `--model` was supplied,
                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(n_inp, n_hid, n_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(n_hid, n_token)

        if tie_weights:
            if n_hid != n_inp:
                raise ValueError('When using tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.n_hid = n_hid
        self.n_layers = n_layers

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def __format__(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, bsz, self.n_hid),
                    weight.new_zeros(self.n_layers, bsz, self.n_hid))
        else:
            return weight.new_zeros(self.n_layers, bsz, self.n_hid)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
            in the sequence. The positional encodings have the same dimension as
            the embeddings, so that the two can be summed. Here, we use sine and cosine
            functions of different frequencies.
        .. math::
            \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
            \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
            \text{where pos is the word position and i is the embed idx)
        Args:
            d_model: the embed dim (required).
            dropout: the dropout value (default=0.1).
            max_len: the max. length of the incoming sequence (default=5000).
        Examples:
        """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def __format__(self, x):
        r"""Inputs of forward function
               Args:
                   x: the sequence fed to the positional encoder model (required).
               Shape:
                   x: [sequence length, batch size, embed dim]
                   output: [sequence length, batch size, embed dim]
               Examples:

               """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, n_token, n_inp, n_head, n_hid, n_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module dose not exist in PyTorch 1.1 or lower')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(n_inp, dropout)
        encoder_layers = TransformerEncoderLayer(n_inp, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_token, n_inp)
        self.n_inp = n_inp
        self.decoder = nn.Linear(n_inp, n_token)

        self.init_weights()

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = .1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.n_inp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
