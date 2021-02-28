import numpy as np
import mxnet as mx
import argparse, math
import logging
from AI.MxnetExamples.RNN.Word_LM.Model import rnn, softmax_ce_loss
from AI.MxnetExamples.RNN.Word_LM.Module import *
from AI.MxnetExamples.RNN.Word_LM.Data import CorpusIter, Corpus
from mxnet.model import BatchEndParam


parser = argparse.ArgumentParser(description='Sherlock Holmes LSTM Language Model')
parser.add_argument('--data', type=str, default='E:/Python_Data/sherlockholmes/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=650,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping by global norm')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
args = parser.parse_args()

best_loss = 9999


def evaluate(valid_module, data_iter, epoch, mode, bptt, batch_size):
    total_loss = 0.0
    n_batch = 0
    for batch in data_iter:
        valid_module.forward(batch, is_train=False)
        outputs = valid_module.get_loss()
        total_loss += mx.nd.sum(outputs[0]).asscalar()
        n_batch += 1
    data_iter.reset()
    loss = total_loss / bptt / batch_size / n_batch
    logging.info('Iter[%d] %s loss:\t%.7f, Perplexity: %.7f' %
                     (epoch, mode, loss, math.exp(loss)))
    return loss


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    args = parser.parse_args()
    logging.info(args)
    ctx = mx.gpu()
    batch_size = args.batch_size
    bptt = args.bptt
    mx.random.seed(args.seed)

    corpus = Corpus(args.data)
    n_tokens = len(corpus.dictionary)
    train_data = CorpusIter(corpus.train, batch_size, bptt)
    valid_data = CorpusIter(corpus.valid, batch_size, bptt)
    test_data = CorpusIter(corpus.test, batch_size, bptt)

    pred, states, state_names = rnn(bptt, n_tokens, args.emsize, args.nhid,
                                    args.nlayers, args.dropout, batch_size, args.tied)

    loss = softmax_ce_loss(pred)

    module = CustomStatefulModule(loss, states, state_names=state_names, context=ctx)
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    module.init_params()
    optimizer = mx.optimizer.create('sgd', learning_rate=args.lr,
                                    rescale_grad=1.0/batch_size)
    module.init_optimizer(optimizer=optimizer)
    speedometer = mx.callback.Speedometer(batch_size, args.log_interval)

    logging.info("Training started...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batch = 0
        for batch in train_data:
            module.forward(batch)
            module.backward()
            module.update(max_norm=args.clip * bptt * batch_size)

            outputs = module.get_loss()
            total_loss += mx.nd.sum(outputs[0]).asscalar()
            speedometer_param = BatchEndParam(epoch=epoch, nbatch=n_batch,
                                              eval_metric=None, locals=locals())
            speedometer(speedometer_param)
            if n_batch % args.log_interval == 0 and n_batch > 0:
                cur_loss = total_loss / bptt / batch_size / args.log_interval
                logging.info('Iter[%d] Batch [%d]\tLoss: %.7f,\tPerplexity: %.7f' %
                             (epoch, n_batch, cur_loss, math.exp(cur_loss)))
                total_loss = 0.0
            n_batch += 1
        valid_loss = evaluate(module, valid_data, epoch, "Valid", bptt, batch_size)
        if valid_loss < best_loss:
            best_loss = valid_loss
            test_loss = evaluate(module, test_data, epoch, "Test", bptt, batch_size)
        else:
            optimizer.lr *= 0.25
        train_data.reset()
    logging.info("Training completed. ")


# parser = argparse.ArgumentParser(description='Sherlock Holmes LSTM Language Model')
# parser.add_argument('--data', type=str, default='E:/Python_Data/sherlockholmes/',
#                     help='location of the data corpus')
# parser.add_argument('--emsize', type=int, default=650,
#                     help='size of word embeddings')
# parser.add_argument('--nhid', type=int, default=650,
#                     help='number of hidden units per layer')
# parser.add_argument('--nlayers', type=int, default=2,
#                     help='number of layers')
# parser.add_argument('--lr', type=float, default=1.0,
#                     help='initial learning rate')
# parser.add_argument('--clip', type=float, default=0.2,
#                     help='gradient clipping by global norm')
# parser.add_argument('--epochs', type=int, default=40,
#                     help='upper epoch limit')
# parser.add_argument('--batch_size', type=int, default=32,
#                     help='batch size')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='dropout applied to layers (0 = no dropout)')
# parser.add_argument('--tied', action='store_true',
#                     help='tie the word embedding and softmax weights')
# parser.add_argument('--bptt', type=int, default=35,
#                     help='sequence length')
# parser.add_argument('--log-interval', type=int, default=200,
#                     help='report interval')
# parser.add_argument('--seed', type=int, default=3,
#                     help='random seed')
# args = parser.parse_args()
#
# best_loss = 9999
#
#
# def evaluate(valid_module, data_iter, epoch, mode, bptt, batch_size):
#     total_loss = 0.0
#     nbatch = 0
#     for batch in data_iter:
#         valid_module.forward(batch, is_train=False)
#         outputs = valid_module.get_loss()
#         total_loss += mx.nd.sum(outputs[0]).asscalar()
#         nbatch += 1
#     data_iter.reset()
#     loss = total_loss / bptt / batch_size / nbatch
#     logging.info('Iter[%d] %s loss:\t%.7f, Perplexity: %.7f' %
#                  (epoch, mode, loss, math.exp(loss)))
#     return loss
#
#
# if __name__ == '__main__':
#     # args
#     head = '%(asctime)-15s %(message)s'
#     logging.basicConfig(level=logging.DEBUG, format=head)
#     args = parser.parse_args()
#     logging.info(args)
#     ctx = mx.gpu()
#     batch_size = args.batch_size
#     bptt = args.bptt
#     mx.random.seed(args.seed)
#
#     # data
#     corpus = Corpus(args.data)
#     ntokens = len(corpus.dictionary)
#     train_data = CorpusIter(corpus.train, batch_size, bptt)
#     valid_data = CorpusIter(corpus.valid, batch_size, bptt)
#     test_data = CorpusIter(corpus.test, batch_size, bptt)
#
#     # model
#     pred, states, state_names = rnn(bptt, ntokens, args.emsize, args.nhid,
#                                     args.nlayers, args.dropout, batch_size, args.tied)
#     loss = softmax_ce_loss(pred)
#
#     # module
#     module = CustomStatefulModule(loss, states, state_names=state_names, context=ctx)
#     module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
#     module.init_params(initializer=mx.init.Xavier())
#     optimizer = mx.optimizer.create('sgd', learning_rate=args.lr, rescale_grad=1.0 / batch_size)
#     module.init_optimizer(optimizer=optimizer)
#
#     # metric
#     speedometer = mx.callback.Speedometer(batch_size, args.log_interval)
#
#     # train
#     logging.info("Training started ... ")
#     for epoch in range(args.epochs):
#         # train
#         total_loss = 0.0
#         nbatch = 0
#         for batch in train_data:
#             module.forward(batch)
#             module.backward()
#             module.update(max_norm=args.clip * bptt * batch_size)
#             # update metric
#             outputs = module.get_loss()
#             total_loss += mx.nd.sum(outputs[0]).asscalar()
#             speedometer_param = BatchEndParam(epoch=epoch, nbatch=nbatch,
#                                               eval_metric=None, locals=locals())
#             speedometer(speedometer_param)
#             if nbatch % args.log_interval == 0 and nbatch > 0:
#                 cur_loss = total_loss / bptt / batch_size / args.log_interval
#                 logging.info('Iter[%d] Batch [%d]\tLoss:  %.7f,\tPerplexity:\t%.7f' % \
#                              (epoch, nbatch, cur_loss, math.exp(cur_loss)))
#                 total_loss = 0.0
#             nbatch += 1
#         # validation
#         valid_loss = evaluate(module, valid_data, epoch, 'Valid', bptt, batch_size)
#         if valid_loss < best_loss:
#             best_loss = valid_loss
#             # test
#             test_loss = evaluate(module, test_data, epoch, 'Test', bptt, batch_size)
#         else:
#             optimizer.lr *= 0.25
#         train_data.reset()
#     logging.info("Training completed. ")
