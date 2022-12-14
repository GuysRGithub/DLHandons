import os
import sys
import mxnet as mx
import logging
import AI.MxnetExamples.StochastisDeep.StochasticDeepModule as SD_module


def get_conv(
        name,
        data,
        num_filter,
        kernel,
        stride,
        pad,
        with_relu,
        bn_momentum):
    conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter,
                                 kernel=kernel, pad=pad, stride=stride, no_bias=True)

    bn = mx.symbol.BatchNorm(name=name + "_bn",
                             data=conv,
                             fix_gamma=False,
                             momentum=bn_momentum,
                             eps=2e-5)

    return mx.symbol.Activation(name=name + "_relu", data=bn, act_type='relu') if with_relu else bn


death_rates = [0.3]
contexts = [mx.context.cpu()]

data = mx.symbol.Variable("data")
conv = get_conv(
    name='conv0',
    data=data,
    num_filter=16,
    kernel=(3, 3),
    stride=(1, 1),
    pad=(1, 1),
    with_relu=True,
    bn_momentum=0.9)

base_mod = mx.module.Module(conv, context=contexts)
mod_seq = mx.mod.SequentialModule()
mod_seq.add(base_mod)

for i in range(len(death_rates)):
    conv = get_conv(
        name='conv0_%d' % i,
        data=mx.sym.Variable("data_%d" % i),
        num_filter=16,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        with_relu=True,
        bn_momentum=0.9)

    conv1 = get_conv(
        name='conv1_%d' % i,
        data=conv,
        num_filter=16,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        with_relu=False,
        bn_momentum=0.9)

    mod = SD_module.StochasticDeepModule(conv, data_names=['data_%d' % i],
                                         context=contexts, death_rate=death_rates[i])
    mod_seq.add(mod, auto_wiring=True)

act = mx.sym.Activation(mx.sym.Variable("data_final"), act_type='relu')
flat = mx.sym.Flatten(act)
pred = mx.sym.FullyConnected(flat, num_hidden=10)
softmax = mx.sym.SoftmaxOutput(pred, name='softmax')
mod_seq.add(mx.mod.Module(softmax, context=contexts, data_names=['data_final']),
            auto_wiring=True, take_labels=True)

n_epoch = 2
batch_size = 100

base_dir = os.path.dirname(__file__)
mx.test_utils.get_mnist_ubyte()

train = mx.io.MNISTIter(
        image=os.path.join(base_dir, "data", "train-images-idx3-ubyte"),
        label=os.path.join(base_dir, "data", "train-labels-idx1-ubyte"),
        input_shape=(1, 28, 28), flat=False,
        batch_size=batch_size, shuffle=True, silent=False, seed=10)
val = mx.io.MNISTIter(
        image=os.path.join(base_dir, "data", "t10k-images-idx3-ubyte"),
        label=os.path.join(base_dir, "data", "t10k-labels-idx1-ubyte"),
        input_shape=(1, 28, 28), flat=False,
        batch_size=batch_size, shuffle=True, silent=False)

logging.basicConfig(level=logging.DEBUG)
mod_seq.fit(train, val, optimizer_params={'learning_rate': 0.01, 'momentum': 0.9},
            num_epoch=n_epoch, batch_end_callback=mx.callback.Speedometer(batch_size, 10))
