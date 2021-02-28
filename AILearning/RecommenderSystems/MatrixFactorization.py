from d2l import AllDeepLearning as d2l
from mxnet import nd, autograd, gluon, init
from mxnet.gluon import nn
import mxnet as mx
from AI.AILearning.RecommenderSystems import MovieLens as loader

"""

RMSE = sqrt(1/|T| * ∑(u,i)∈T(Rui−R^ui)2)

"""


class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_users, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + nd.squeeze(b_u) + nd.squeeze(b_i)
        return outputs.flatten()


def evaluator(net, test_iter, ctx):
    rmse = mx.metric.RMSE()
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, ctx, even_split=False)
        i = gluon.utils.split_and_load(items, ctx, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, ctx, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(nd.mean(nd.array(rmse_list)).asscalar())


def train_recsys_rating(net, train_iter, test_iter, loss, trainer,
                        num_epochs, ctx_list=d2l.try_all_gpus(),
                        evaluator=None, **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0
        for i, values in enumerate(train_iter):
            if i == 2:
                print(values)
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, ctx_list))
            train_feat = input_data[0:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum(l.asnumpy() for l in ls).mean() / len(ctx_list)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'], ctx_list)
        else:
            test_rmse = evaluator(net, test_iter, ctx_list)
        train_l = 1 / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print('train loss %.3f, test RMSE %.3f ' % (metric[0] / metric[1], test_rmse))


# ctx = d2l.try_all_gpus()
# num_users, num_items, train_iter, test_iter = loader.split_and_load_ml100k(test_ratio=0.1,
#                                                                            batch_size=512)
# net = MF(30, num_users, num_items)
# net.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
# lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
# loss = gluon.loss.L2Loss()
# trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer, optimizer_params={"learning_rate": lr, 'wd': wd})
# train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs, ctx, evaluator)



