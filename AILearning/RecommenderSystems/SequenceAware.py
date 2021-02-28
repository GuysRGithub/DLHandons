from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, init
from mxnet import nd as np
from mxnet.gluon import nn
import random
import sys
from AI.AILearning.RecommenderSystems import MatrixFactorization as helper
from AI.AILearning.RecommenderSystems import PersonalizedRandking as helper_loss
import AI.AILearning.RecommenderSystems.NeuralCollaborativeFiltering as helper_train

"""
    ###################         CASER      MODEL     ############################
    short for convolutional sequence embedding recommendation model,
    adopts convolutional neural networks capture the dynamic pattern
    influences of users’ recent activities. The main component of Caser
    consists of a horizontal convolutional network and a vertical convolutional
    network, aiming to uncover the union-level and point-level sequence patterns,
    respectively. Point-level pattern indicates the impact of single item in the
    historical sequence on the target item, while union level pattern implies the
    influences of several previous actions on the subsequent target. For example,
    buying both milk and butter together leads to higher probability of buying flour
    than just buying one of them. Moreover, users’ general interests, or long term
    preferences are also modeled in the last fully-connected layers, resulting in a more
    comprehensive modeling of user interests. Details of the model are described as follows.

     ###############################3        ARCHITECTURES MODEL   ###########################
     In sequence-aware recommendation system, each user is associated with a sequence of
     some items from the item set. Let  Su=(Su1,...Su|Su|)  denotes the ordered sequence.
     The goal of Caser is to recommend item by considering user general tastes as well as
     short-term intention. Suppose we take the previous  L  items into consideration,
     an embedding matrix that represents the former interactions for timestep  t  can be
     constructed:

    E(u,t)=[qSut−L,...,qSut−2,qSut−1]⊤,

    where  Q∈Rn×k  represents item embeddings and  qi  denotes the  ith  row.
    E(u,t)∈RL×k  can be used to infer the transient interest of user  u
    at time-step  t . We can view the input matrix  E(u,t)  as an image which is
    the input of the subsequent two convolutional components.

    The horizontal convolutional layer has  d  horizontal filters
    Fj∈Rh×k,1≤j≤d,h={1,...,L} , and the vertical convolutional layer has
    d′  vertical filters  Gj∈RL×1,1≤j≤d′ . After a series of convolutional and pool
    operations, we get the two outputs:

    o=HConv(E(u,t),F)
    o′=VConv(E(u,t),G),

    where  o∈Rd  is the output of horizontal convolutional network and
    o′∈Rkd′  is the output of vertical convolutional network. For simplicity,
    we omit the details of convolution and pool operations. They are concatenated
    and fed into a fully-connected neural network layer to get more high-level representations.

    z=ϕ(W[o,o′].⊤+b),

    where  W∈Rk×(d+kd′)  is the weight matrix and  b∈Rk  is the bias.
    The learned vector  z∈Rk  is the representation of user’s short-term intent.

    At last, the prediction function combines users’ short-term and general
    taste together, which is defined as:

    y^uit=vi⋅[z,pu]⊤+b′i,

    where  V∈Rn×2k  is another item embedding matrix.  b′∈Rn
    is the item specific bias.  P∈Rm×k  is the user embedding matrix for users’
    general tastes.  pu∈Rk  is the  uth  row of  P  and  vi∈R2k  is the  ith  row of  V .

"""


class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L, units=num_factors,
                           activation='relu')
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.drop_out = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = nd.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape()[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = nd.squeeze(nd.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = nd.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = nd.concat(*out_hs, dim=1)
        out = nd.concat(*[out_v, out_h], dim=1)
        z = self.fc(self.drop_out(out))
        x = nd.concat(*[z, user_emb], dim=1)
        q_prime_i = nd.squeeze(self.Q_prime(item_id))
        b = nd.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res


class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items, candidates):
        user_ids, item_ids = nd.array(user_ids), nd.array(item_ids)
        sort_idx = nd.array(sorted(range(len(user_ids)), key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = nd.array([i[0] for i in temp])
        idx = nd.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1
                                for c in nd.array([len(i[1]) for i in temp])]).asscalar())
        self.seq_items = nd.zeros((ns, L))
        self.seq_users = nd.zeros(ns, dtype='int32')
        self.seq_tgt = nd.zeros((ns, 1))
        self.test_seq = nd.zeros((num_users, L))
        test_users, _uid = nd.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, -step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1].asscalar())
            for s in self._win(i_ids[int(idx[i].asscalar()):stop_idx], max_len):
                yield int(u_ids[i].asscalar()), s

    def __len__(self):
        return self.ns

    def __getitem__(self, item):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[item].asscalar())]))
        idx = random.randint(0, len(neg) - 1)
        return self.seq_users[item], self.seq_items[item], \
            self.seq_tgt[item], neg[idx]


TARGET_NUM, L, batch_size = 1, 3, 4096
df, num_users, num_items = helper.loader.read_data_ml100k()
train_data, test_data = helper.loader.split_data_ml100k(df, num_users, num_items,
                                                        split_mode='seq-aware')
users_train, items_train, ratings_train, candidates = \
    helper.loader.load_data_ml100k(train_data, num_users, num_items, feedback='implicit')
users_test, items_test, ratings_test, test_iter = \
    helper.loader.load_data_ml100k(test_data, num_users, num_items, feedback='implicit')
train_seq_data = SeqDataset(users_train, items_train, L, num_users, num_items, candidates)
num_workers = 0 if sys.platform.startswith('win') else 4
train_iter = gluon.data.DataLoader(train_seq_data, batch_size,
                                   True, last_batch='rollover',
                                   num_workers=num_workers)
test_seq_iter = train_seq_data.test_seq

"""
    The training data structure is shown above. The first element is the user identity, 
    the next list indicates the last five items this user liked, and the last 
    element is the item this user liked after the five items.
"""

ctx = d2l.try_all_gpus()
net = Caser(10, num_users, L)
net.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = helper_loss.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer, {"learning_rate": lr, 'wd': wd})
helper_train.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                           num_users, num_items, num_epochs, ctx,
                           helper_train.evaluate_ranking, candidates, eval_step=1)
d2l.plt.show()

"""

#######################                SUMMARY          #############################

    Inferring a user’s short-term and long-term interests can make
     prediction of the next item that she preferred more effectively.

    Convolutional neural networks can be utilized to capture users’ 
    short-term interests from sequential interactions.
    
"""