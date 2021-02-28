import mxnet as mx
import numpy as np


def rse(label, pred):
    """
    computes the root relative squared error (condensed using standard deviation formula)
    :param label:
    :param pred:
    :return: squared error
    """
    numerator = np.sqrt(np.mean(np.square(label - pred), axis=None))
    denominator = np.std(label, axis=None)
    return numerator / denominator


def rae(label, pred):
    """
    computes the relative absolute error (condensed using standard deviation formula)
    :param label:
    :param pred:
    :return:
    """
    numerator = np.mean(np.abs(label - pred), axis=None)
    denominator = np.mean(np.abs(label - np.mean(label, axis=None)), axis=None)
    return numerator / denominator


def corr(label, pred):
    """
    computes the empirical correlation coefficient
    :param label:
    :param pred:
    :return:
    """
    numerator1 = label - np.mean(label, axis=0)
    numerator2 = pred - np.mean(pred, axis=0)
    numerator = np.mean(numerator1 * numerator2, axis=0)
    denominator = np.std(label, axis=0) * np.std(pred, axis=0)
    return numerator / denominator


def get_custom_metrics():
    _rse = mx.metric.create(rse)
    _rae = mx.metric.create(rae)
    _corr = mx.metric.create(corr)
    return mx.metric.create([_rae, _rse, _corr])


def evaluate(pred, label):
    return {'RAE': rae(label, pred), 'RSE': rse(label, pred), 'CORR': corr(label, pred)}

