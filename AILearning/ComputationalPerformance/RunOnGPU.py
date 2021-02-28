import numpy as np
from timeit import default_timer as timer
from mxnet import nd
import mxnet as mx


def pow(a, b):
    return a ** b


def main():
    vec_size = 1000000

    a = b = nd.array(np.random.sample(vec_size), dtype=np.float32, ctx=mx.gpu())

    start = timer()
    c = pow(a, b)
    duration = timer() - start
    print(a.context)
    print(duration)


if __name__ == '__main__':
    main()
