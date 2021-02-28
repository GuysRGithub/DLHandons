from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, init
from mxnet.gluon import nn
import sys
from AI.AILearning.RecommenderSystems import FeatureRich as helper
from AI.AILearning.RecommenderSystems import MatrixFactorization as train_helper
"""
    ###################         DEEP FM            ###############

    DeepFM consists of an FM component and a deep component which
    are integrated in a parallel structure. The FM component is the
    same as the 2-way factorization machines which is used to model the
    low-order feature interactions. The deep component is a multi-layered
    perceptron that is used to capture high-order feature interactions and
    nonlinearities. These two components share the same inputs/embeddings and
    their outputs are summed up as the final prediction. It is worth pointing out that
    the spirit of DeepFM resembles that of the Wide & Deep architecture which can
    capture both memorization and generalization. The advantages of DeepFM over
    the Wide & Deep model is that it reduces the effort of hand-crafted feature
    engineering by identifying feature combinations automatically
    
    Let  ei∈Rk  denote the latent feature vector of the  ith  field. 
    The input of the deep component is the concatenation of the dense embeddings 
    of all fields that are looked up with the sparse categorical feature input, denoted as:

    z(0)=[e1,e2,...,ef],
 
    where  f  is the number of fields. It is then fed into the following neural network:

    z(l)=α(W(l)z(l−1)+b(l)),
 
    where  α  is the activation function.  Wl  and  bl  are the weight and bias at the 
     lth  layer. Let  yDNN  denote the output of the prediction. The ultimate prediction
     of DeepFM is the summation of the outputs from both FM and DNN. So we have:

    y^=σ(y^(FM)+y^(DNN)),
 
    where  σ  is the sigmoid function. The architecture of DeepFM is illustrated below.
"""


class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        input_size = int(sum(field_dims).asscalar())
        self.embedding = nn.Embedding(input_size, num_factors)
        self.fc = nn.Embedding(input_size, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', in_units=input_dim))
            self.mlp.add(nn.Dropout(drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = nd.sum(embed_x, axis=1) ** 2
        sum_of_square = nd.sum(embed_x ** 2, axis=1)
        inputs = nd.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) + 0.5 * \
            (square_of_sum - sum_of_square).sum(1, keepdims=True) + self.mlp(inputs)
        x = nd.sigmoid(x)
        return x


batch_size = 2048
data_dir = "E:/Python_Data/ctr/ctr/"
train_data = helper.CTRDataset(data_dir + "train.csv")
test_data = helper.CTRDataset(data_dir + "test.csv", train_data.feat_mapper,
                              train_data.defaults)
fields_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(train_data, shuffle=True,
                                   last_batch="rollover",
                                   batch_size=batch_size,
                                   num_workers=0)
test_iter = gluon.data.DataLoader(test_data, shuffle=True,
                                  last_batch="rollover",
                                  batch_size=batch_size,
                                  num_workers=0)

ctx = d2l.try_all_gpus()
net = DeepFM(fields_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=ctx)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer, {"learning_rate": lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch12(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)

d2l.plt.show()


"""
        ##########################        SUMMARY              ##########################
        Integrating neural networks to FM enables it to model complex and high-order 
        interactions.

        DeepFM outperforms the original FM on the advertising dataset.
        
"""