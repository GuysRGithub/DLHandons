import numpy as np
import mxnet as mx


def get_mnist():
    """

    :return: Mnist Data concatenated
    """
    np.random.seed(1234)
    mnist_data = mx.test_utils.get_mnist()
    X = np.concatenate([mnist_data['train_data'], mnist_data['test_data']])
    Y = np.concatenate([mnist_data['train_label'], mnist_data['test_label']])
    p = np.random.permutation(X.shape[0])
    X = X[p].reshape((X.shape[0], -1)).astype(np.float32) * 5
    Y = Y[p]
    return X, Y

