import argparse
import logging
import mxnet as mx
from AI.MxnetExamples.MatrixFactorization.GetData import get_movie_lens_iter
from AI.MxnetExamples.MatrixFactorization.Model import matrix_fact_model_parallel_net

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run model parallel version"
                                             "of matrix factorization")
parser.add_argument('--num-epoch', type=int, default=3,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256,
                    help='number of examples per batch')
parser.add_argument('--print-every', type=int, default=100,
                    help='logging interval')
parser.add_argument('--factor-size', type=int, default=128,
                    help="the factor size of the embedding operation")
parser.add_argument('--num-gpus', type=int, default=1,
                    help="number of gpus to use")

MOVIELENS = {
    'dataset': 'ml-10m',
    'train': 'ml-10m/ml-10M100K/r1.train',
    'val': 'ml-10m/ml-10M100K/r1.test',
    'max_user': 71569,
    'max_movie': 65135,
}

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    args = parser.parse_args()
    logging.info(args)
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    optimizer = 'sgd'
    factor_size = args.factor_size
    print_every = args.print_every
    num_gpus = args.num_gpus

    momentum = 0.9
    learning_rate = 0.1

    max_user = MOVIELENS['max_user']
    max_movies = MOVIELENS['max_movie']
    train_iter = get_movie_lens_iter(MOVIELENS['train'], batch_size)
    val_iter = get_movie_lens_iter(MOVIELENS['val'], batch_size)

    net = matrix_fact_model_parallel_net(factor_size, factor_size, max_user, max_movies)
    group2ctx = {'dev1': [mx.cpu()] * num_gpus, 'dev2': [mx.gpu(i) for i in range(num_gpus)]}
    mod = mx.module.Module(symbol=net, context=[mx.cpu(), mx.gpu()],
                           data_names=['user', 'item'],
                           label_names=['score'], group2ctxs=group2ctx)
    initializer = mx.init.Xavier(factor_type='in', magnitude=2.34)

    optimizer_params = {
        'learning_rate': learning_rate,
        'wd': 1e-4,
        'momentum': momentum,
        'rescale_grad': 1.0 / batch_size
    }

    metric = mx.metric.create(['MSE'])

    mod.fit(train_iter,
            val_iter,
            eval_metric=metric,
            num_epoch=num_epoch,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            initializer=initializer,
            batch_end_callback=mx.callback.Speedometer(batch_size, print_every))
