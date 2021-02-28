import os
import logging
import argparse
import numpy as np
import mxnet as mx
import AI.MxnetExamples.TextCNN.DataHelper as data_helper

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="CNN for text classification")
parser.add_argument('--pretrained-embedding', action='store_true',
                    help='use pre-trained word2vec only if specified')
parser.add_argument('--num-embed', type=int, default=300,
                    help='embedding layer size')
parser.add_argument('--gpus', type=str, default='0',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')
parser.add_argument('--kv-store', type=str, default='local',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=200,
                    help='max num of epochs')
parser.add_argument('--batch-size', type=int, default=50,
                    help='the batch size.')
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    help='the optimizer type')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout rate')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
parser.add_argument('--save-period', type=int, default=10,
                    help='save checkpoint for every n epochs')

args = parser.parse_args()


def save_model():
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    return mx.callback.do_checkpoint("checkpoint/checkpoint", args.sabe_period)


def data_iter(batch_size, num_embed, pre_trained_word2vec=False):
    print("Loading data...")
    if pre_trained_word2vec:
        word2vec = data_helper.load_pretrained_word2vec("data/re.vec")
        x, y = data_helper.load_data_with_word2vec(word2vec)
        x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        embedded_size = x.shape[-1]
        sentences_size = x.shape[2]
        vocabulary_size = -1
    else:
        x, y, vocab, vocab_inv = data_helper.load_data()
        embedded_size = num_embed
        sentences_size = x.shape[1]
        vocabulary_size = len(vocab)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
    print("Train/Valid split: %d/%d" % (len(y_train), len(y_dev)))
    print("Train shape: ", x_train.shape)
    print("Valid shape: ", x_dev.shape)
    print("Sentence max words", sentences_size)
    print("Embedding size", embedded_size)
    print("Vocab size", vocabulary_size)

    train_set = mx.io.NDArrayIter(x_train, y_train, batch_size, shuffle=True)
    valid = mx.io.NDArrayIter(x_dev, y_dev, batch_size)

    return train_set, valid, sentences_size, embedded_size, vocabulary_size


def sym_gem(batch_size, sentences_size, num_embed, vocabulary_size,
            num_label=2, filter_list=None, num_filter=100,
            dropout=0.0, pre_trained_word2vec=False):
    """

    :param batch_size:
    :param sentences_size:
    :param num_embed:
    :param vocabulary_size:
    :param num_label:
    :param filter_list:
    :param num_filter:
    :param dropout:
    :param pre_trained_word2vec:
    :return:
    """

    input_x = mx.sym.Variable("data")
    input_y = mx.sym.Variable("softmax_label")

    if not pre_trained_word2vec:
        embed_layer = mx.sym.Embedding(data=input_x,
                                       input_dim=vocabulary_size,
                                       output_dim=num_embed,
                                       name="vocab_embed")
        conv_input = mx.sym.Reshape(data=embed_layer,
                                    target_shape=(batch_size, 1, sentences_size, num_embed))
    else:
        conv_input = input_x

    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        conv_i = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed),
                                    num_filter=num_filter)
        relu_i = mx.sym.Activation(data=conv_i, act_type='relu')
        pool_i = mx.sym.Pooling(data=relu_i, pool_type='max',
                                kernel=(sentences_size - filter_size + 1, 1),
                                stride=1)
        pooled_outputs.append(pool_i)

    total_filters = num_filter * len(filter_list)
    concat = mx.sym.concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))

    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")

    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias,
                               num_hidden=num_label)

    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm, ('data',), ('softmax_label',)


def train(symbol_data, train_iterator, valid_iterator, data_column_names,
          target_names):
    """

    :param symbol_data:
    :param train_iterator:
    :param valid_iterator:
    :param data_column_names:
    :param target_names:
    :return:
    """

    devs = mx.cpu()
    if args.gpus is not None:
        for i in args.gpus.split(','):
            mx.gpu(int(i))
        devs = mx.gpu()

    module = mx.mod.Module(symbol_data, data_names=data_column_names,
                           label_names=target_names, context=devs)
    module.fit(train_data=train_iterator,
               eval_data=valid_iterator,
               eval_metric='acc',
               optimizer=args.optimizer,
               optimizer_params={'learning_rate': args.lr},
               initializer=mx.initializer.Uniform(0.1),
               num_epoch=args.num_epoch,
               batch_end_callback=mx.callback.Speedometer(args.batch_size, args.display_batch),
               epoch_end_callback=save_model())


if __name__ == '__main__':
    train_ier, valid_iter, sentences_size, embed_size, vocab_size = \
        data_iter(args.batch_size, args.num_embed, args.pretrained_embedding)
    symbol, data_names, target_names = sym_gem(args.batch_size,
                                               sentences_size,
                                               embed_size,
                                               vocab_size,
                                               num_label=2,
                                               filter_list=[3, 4, 5],
                                               num_filter=100,
                                               dropout=args.dropout,
                                               pre_trained_word2vec=args.pretrained_embedding)
    train(symbol, train_ier, valid_iter, data_names, target_names)
