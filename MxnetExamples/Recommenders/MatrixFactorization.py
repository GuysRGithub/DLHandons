import logging
import math
import random
import mxnet as mx
from mxnet import gluon, autograd
import numpy as np
from d2l import AllDeepLearning as d2l

logging.basicConfig(level=logging.DEBUG)


def evaluate_network(network, data_iterator, ctx):
    loss_acc = .0
    l2 = gluon.loss.L2Loss()
    idx = 0
    for idx, (users, items, scores) in enumerate(data_iterator):
        users_ = gluon.utils.split_and_load(users, ctx)  # split data into multi gpu
        items_ = gluon.utils.split_and_load(items, ctx)
        scores_ = gluon.utils.split_and_load(scores, ctx)
        preds = [network(u, i) for u, i in zip(users_, items_)]
        losses = [l2(p, s).asnumpy() for p, s in zip(preds, scores_)]
        loss_acc += sum(losses).mean() / len(ctx)
    return loss_acc / (idx + 1)


def train(network, train_data, test_data, epochs, lr=0.1, optimizer='sgd',
          ctx=d2l.try_all_gpus(), num_epoch_lr=5, factor=0.2):
    np.random.seed(123)
    mx.random.seed(123)
    random.seed(123)

    schedule = mx.lr_scheduler.FactorScheduler(step=len(train_data) * len(ctx) * num_epoch_lr,
                                               factor=factor)
    trainer = gluon.Trainer(network.collect_params(), optimizer,
                            {"learning_rate": lr, 'lr_scheduler': schedule})
    l2 = gluon.loss.L2Loss()

    network.hybridize()
    idx = 0
    losses_output = []
    for epoch in range(epochs):
        loss_acc = 0
        for idx, (users, items, scores) in enumerate(train_data):

            users_ = gluon.utils.split_and_load(users, ctx)
            items_ = gluon.utils.split_and_load(items, ctx)
            scores_ = gluon.utils.split_and_load(scores, ctx)

            with autograd.record():
                preds = [network(u, i) for u, i in zip(users_, items_)]
                losses = [l2(p, s) for p, s in zip(preds, scores_)]

            [l.backward() for l in losses]
            loss_acc += sum([l.asnumpy() for l in losses]).mean() / len(ctx)
            trainer.update(users.shape[0])

        test_loss = evaluate_network(network, train_data, ctx)
        train_loss = loss_acc / (idx + 1)
        print('epoch [{}], training rmse {}:.4f, test rmse {:.4f}'
              .format(epoch, train_loss, test_loss))
        losses_output.append((train_loss, test_loss))
    return losses_output


