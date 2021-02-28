from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, init
from mxnet.gluon import nn
import sys
from AI.AILearning.RecommenderSystems import FeatureRich as loader

data_dir = "E:/Python_Data/ctr/ctr/"

"""
    #################################       2-way factorization machines               ###########################
    The model for a factorization machine of degree two is defined as:
    y^(x)=w0+∑i=1->d (w_i * x_i)+∑i=1->d∑j=i+1->d⟨vi,vj⟩x_i * x_j
    where  w0∈R  is the global bias;  w∈Rd  denotes the weights of the i-th
    variable;  V∈Rd×k  represents the feature embeddings;  vi  represents the
    ith  row of  V ;  k  is the dimensionality of latent factors;  ⟨⋅,⋅⟩  is the
    dot product of two vectors.  ⟨vi,vj⟩  model the interaction between the  ith
    and  jth  feature.

    The reformulation of the pairwise interaction term is as follows:
        ∑i=1->d∑j=i+1->d⟨vi,vj⟩x_ix_j
            =1/2 * ∑i=1->d∑j=1->d⟨vi,vj⟩x_ix_j − 1/2 * ∑i=1->d⟨vi,vi⟩x_ix_i
            =1/2 (∑i=1->d∑j=1->d∑l=1->k v_(i,l) v_(j,l)x_x_j − ∑i=1->d∑l=1->k v_(i,l)v_(j,l)x_ix_i)
            =1/2 ∑l=1->k((∑i=1->d v_(i,l)x_i)(∑j=1->d v_(j,l)x_j) − ∑i=1->d (v_(i,l))^2,x_i^2)
            =1/2 ∑l=1->k((∑i=1->d v_(i,l)x_i)^2 − ∑i=1->d v_(i,l)^2 * xi^2)
            
"""


class FM(nn.Block):
    def __init__(self, field_dims, num_factors, **kwargs):
        super(FM, self).__init__(**kwargs)
        input_size = int(sum(field_dims).asscalar())
        self.embedding = nn.Embedding(input_size, num_factors)
        self.fc = nn.Embedding(input_size, 1)
        self.linear_layer = gluon.nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = nd.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = nd.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1) +
                              0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True))
        x = nd.sigmoid(x)
        return x


# batch_size = 2048
# train_data = loader.CTRDataset(data_dir + "train.csv")
# test_data = loader.CTRDataset(data_dir + "test.csv", feat_mapper=train_data.feat_mapper,
#                               defaults=train_data.defaults)
# train_iter = gluon.data.DataLoader(train_data, shuffle=True, last_batch='rollover',
#                                    batch_size=batch_size, num_workers=0)
# test_iter = gluon.data.DataLoader(test_data, shuffle=False, last_batch='rollover',
#                                   batch_size=batch_size, num_workers=0)
batch_size = 2048
train_data = loader.CTRDataset(data_dir + "train.csv")
test_data = loader.CTRDataset(data_dir + "test.csv",
                              feat_mapper=train_data.feat_mapper,
                              defaults=train_data.defaults)
num_workers = 0 if sys.platform.startswith("win") else 4
print(train_data[0])
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch="rollover", batch_size=batch_size,
    num_workers=num_workers)
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch="rollover", batch_size=batch_size,
    num_workers=num_workers)

ctx = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=ctx)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch12(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
# ctx = d2l.try_all_gpus()
# net = FM(train_data.field_dims, num_factors=20)
# net.initialize(init.Xavier(), ctx=ctx)
# lr, num_epochs, optimizer = 0.02, 30, 'adam'
# trainer = gluon.Trainer(net.collect_params(), optimizer,
#                         {'learning_rate': lr})
# loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
# d2l.train_ch12(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
d2l.plt.show()
