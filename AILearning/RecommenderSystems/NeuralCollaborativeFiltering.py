from d2l import AllDeepLearning as d2l
from mxnet import autograd, gluon, nd, init
from mxnet.gluon import nn
import mxnet as mx
import sys
import random
from AI.AILearning.RecommenderSystems import PersonalizedRandking, MovieLens as loader

"""
##############################33                        THE NeuMF MODEL             #############################

    NeuMF fuses two subnetworks. The GMF is a generic neural network version of matrix
    factorization where the input is the elementwise product of user and item
    latent factors. It consists of two neural layers:

    x=pu⊙qi
    y^ui=α(h⊤x),

    where  ⊙  denotes the Hadamard product of vectors
      pu∈Rk  is the  uth  row of  P  and  qi∈Rk  is the  ith  row of  Q .  α
      and  h  denote the activation function and weight of the output layer.
        y^ui  is the prediction score of the user  u  might give to the item  i .

    Another component of this model is MLP. To enrich model flexibility, the MLP
    subnetwork does not share user and item embeddings with GMF

    NeuMF concatenates the second last layers of two subnetworks to create a
    feature vector which can be passed to the further layers. Afterwards, the
    ouputs are projected with matrix  h  and a sigmoid activation function.
    The prediction layer is formulated as:
        y^ui=σ(h⊤[x,ϕL(z(L))]).



"""


class NeuMF(nn.Block):
    def __init__(self, num_factors, num_users,
                 num_items, mlp_layers, **kwargs):
        super(NeuMF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_users, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_users, num_factors)

        self.mlp = nn.Sequential()
        for i in mlp_layers:
            self.mlp.add(nn.Dense(i, activation='relu',
                                  use_bias=True))

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(nd.concat(*[p_mlp, q_mlp], dim=1))
        con_res = nd.concat(*[gmf, mlp], dim=1)
        return nd.sum(con_res, axis=-1)


class PRDataset(gluon.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]


"""

###################    Hit@ℓ=1/m * ∑u∈U1(rank_(u,gu) <= ℓ),            #########################
 
    where  1  denotes an indicator function that is equal to one if the 
    ground truth item is ranked in the top  ℓ  list, 
    otherwise it is equal to zero.  ranku,gu  denotes the ranking of the 
    ground truth item  gu  of the user  u  in the recommendation list (The ideal ranking is 1). 
    m  is the number of users.  U  is the user set.

    The definition of AUC is as follows:

##########    AUC= 1/m * ∑u∈U1|I∖Su|∑j∈I∖Su 1 * (rank_(u,g_u) < rank_(u,j),  ##############################3       

 
    where  I  is the item set.  Su  is the candidate items of user  u . 
    Note that many other evaluation protocols such as precision, 
    recall and normalized discounted cumulative gain (NDCG) can also be used.


"""


def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc


def evaluate_ranking(net, test_input, seq, candidates, num_users,
                     num_items, ctx):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_users)])
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, x, scores = [], [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        x.extend([nd.array(user_ids)])
        if seq is not None:
            x.append(seq[user_ids, :])
        x.extend([nd.array(item_ids)])
        test_data_iter = gluon.data.DataLoader(
            gluon.data.ArrayDataset(*x),
            shuffle=False,
            last_batch='keep',
            batch_size=1024)
        for index, values in enumerate(test_data_iter):
            x = [gluon.utils.split_and_load(v, ctx, even_split=False) for v in values]
            scores.extend([list(net(*t).asnumpy()) for t in zip(*x)])
        scores = [item for sublist in scores for item in sublist]
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return nd.mean(nd.array(hit_rate)), nd.mean(nd.array(auc))


def train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, ctx_list, evaluator,
                  candidates, eval_step=1):
    timer, hit_rate, auc = d2l.Timer(), 0, 0
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            ylim=[0, 1], legend=['test hit rate', 'test AUC'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0
        for i, values in enumerate(train_iter):
            input_data = []
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, ctx_list))
            with autograd.record():
                p_pos = [net(*t) for t in zip(*input_data[0:-1])]
                p_neg = [net(*t) for t in zip(*input_data[0:-2],
                                              input_data[-1])]
                ls = [loss(p, n) for p, n in zip(p_pos, p_neg)]
            [l.backward(retain_graph=False) for l in ls]
            l += sum(l.asnumpy() for l in ls).mean() / len(ctx_list)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        with autograd.record():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter, candidates,
                                          num_users, num_items, ctx_list)
                animator.add(epoch + 1, (hit_rate.asscalar(), auc.asscalar()))
    print('train loss %.3f, test hit rate %.3f, test AUC %.3f' %
          (metric[0] / metric[1], hit_rate.asscalar(), auc.asscalar()))
    print('%.1f examples/sec on %s' % (metric[2] * num_epochs / timer.sum(), ctx_list))


batch_size = 1024
df, num_users, num_items = loader.read_data_ml100k()
train_data, test_data = loader.split_data_ml100k(df, num_users, num_items, 'seq-aware')

users_train, items_train, ratings_train, candidates = \
    loader.load_data_ml100k(train_data, num_users, num_items, feedback='implicit')
users_test, items_test, ratings_test, test_iter = \
    loader.load_data_ml100k(test_data, num_users, num_items, 'implicit')
num_workers = 0 if sys.platform.startswith('win') else 4
train_iter = gluon.data.DataLoader(PRDataset(users_train, items_train, candidates, num_items),
                                   batch_size, True, last_batch='rollover',
                                   num_workers=num_workers)
ctx = d2l.try_all_gpus()
net = NeuMF(10, num_users, num_items, mlp_layers=[10, 10, 10])
net.initialize(init=init.Normal(0.01), ctx=ctx, force_reinit=True)
lr, num_epochs, wd, optimizer = 0.01, 10, 1e-5, 'adam'
loss = PersonalizedRandking.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': lr, 'wd': wd})
train_ranking(net, train_iter, test_iter, loss, trainer, None,
              num_users, num_items, num_epochs, ctx, evaluate_ranking, candidates)
d2l.plt.show()

"""
    ###################                   SUMMARY                 #####################
    Adding nonlinearity to matrix factorization model is beneficial for improving 
    the model capability and effectiveness.

    NeuMF is a combination of matrix factorization and Multilayer perceptron.
    The multilayer perceptron takes the concatenation of user and item embeddings as the input.
"""
