import mxnet as mx
from mxnet.test_utils import get_mnist_iterator
import numpy as np
import logging


class Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        self.assign(in_grad[0], req[0], mx.nd.array(y))


@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Softmax()


data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data=fc1, act_type='relu', name='act1')
fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
act2 = mx.symbol.Activation(data=fc1, name='act2', act_type='relu')
fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)

mlp = mx.symbol.Custom(data=fc3, name="softmax", op_type='softmax')

logging.basicConfig(level=logging.DEBUG)

train, val = get_mnist_iterator(batch_size=100, input_shape=(784, ))

context = mx.cpu()

mod = mx.mod.Module(mlp, context=context)

mod.fit(train_data=train, eval_data=val, optimizer='sgd',
        optimizer_params={'learning_rate': .1, 'momentum': .9, 'wd': 0.00001},
        num_epoch=12, batch_end_callback=mx.callback.Speedometer(100, 100))



