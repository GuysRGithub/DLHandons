from d2l import AllDeepLearning as d2l
from mxnet import autograd, gluon, nd
from mxnet import nd as np
from mxnet.gluon import nn
import mxnet as mx
import sys
from AI.AILearning.RecommenderSystems import MovieLens as loader
from AI.AILearning.RecommenderSystems import MatrixFactorization as train_helper


"""
///////////         AutoRECT            /////////////////// In AutoRec, instead of explicitly embedding 
    users/items into low-dimensional space, it uses the column/row of the interaction matrix as the input, 
    then reconstructs the interaction matrix in the output layer. h(R∗i)=f(W⋅g(VR∗i+μ)+b) Mini: arg_minW,V,μ,
    b∑i=1M∥R∗i−h(R∗i)∥2O+λ(∥W∥2F+∥V∥2F) ///////////////////////                    MODEL                  
////////////////////////// 
    A typical autoencoder consists of an encoder and a decoder. The encoder projects the input 
    to hidden representations and the decoder maps the hidden layer to the reconstruction layer.
     
"""


class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout_rate=0.05):
        super(AutoRec, self).__init__()
        self.encoder = gluon.nn.Dense(num_hidden, activation='sigmoid',
                                      use_bias=True)
        self.decoder = gluon.nn.Dense(num_users, use_bias=True)
        self.dropout_layer = gluon.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        hidden = self.dropout_layer(self.encoder(inputs))
        pred = self.decoder(hidden)
        if autograd.is_training():
            return pred * nd.sign(inputs)
        else:
            return pred


def evaluator(net, inter_matrix, test_data, ctx):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, ctx, even_split=False)
        scores.extend([net(i).asnumpy() for i in feat])
    recons = nd.array([item for sublist in scores for item in sublist])
    rmse = nd.sqrt(nd.sum(nd.square(test_data - nd.sign(test_data) * recons)) /
                   nd.sum(nd.sign(test_data)))
    return float(rmse.asscalar())


ctx = d2l.try_all_gpus()
df, num_users, num_items = loader.read_data_ml100k()
train_data, test_data = loader.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = loader.load_data_ml100k(train_data, num_users, num_items)
_, _, _, test_inter_mat = loader.load_data_ml100k(test_data, num_users, num_items)
num_workers = 0 if sys.platform.startswith('win') else 4
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch='rollover', batch_size=256,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(nd.array(train_inter_mat), shuffle=False,
                                  last_batch='keep', batch_size=1024,
                                  num_workers=num_workers)
for values in train_iter:
    print(values)
    break

net = AutoRec(500, num_users)
net.initialize(ctx=ctx, force_reinit=True,
               init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': lr, 'wd': wd})
train_helper.train_recsys_rating(net, train_iter, test_iter, loss, trainer,
                                 num_epochs, ctx, evaluator, inter_mat=test_inter_mat)
d2l.plt.show()

"""        /////////////           SUMMARY                 ////////////////////////
    We can frame the matrix factorization algorithm with autoencoders,
     while integrating non-linear layers and dropout regularization.

    Experiments on the MovieLens 100K dataset show that AutoRec achieves superior performance than matrix factorization.
"""
