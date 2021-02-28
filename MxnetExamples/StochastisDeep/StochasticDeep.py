import os
import sys
import mxnet as mx
import logging
from AI.MxnetExamples.StochastisDeep import StochasticDeepModule as sd_module


def residual_module(death_rate, n_channel, name_scope, context,
                    stride=1, bn_momentum=0.9):
    data = mx.sym.Variable("data")

    bn1 = mx.symbol.BatchNorm(data=data, name=name_scope + "_bn1", fix_gamma=False,
                              momentum=bn_momentum, eps=2e-5)

    relu1 = mx.symbol.Activation(data=bn1, act_type='relu', name=name_scope + "_relu1")
    conv1 = mx.symbol.Convolution(data=relu1, num_filter=n_channel, kernel=(3, 3),
                                  pad=(1, 1), stride=(stride, stride), name=name_scope + "conv1")

    bn2 = mx.symbol.BatchNorm(data=conv1, name=name_scope + "_bn2", fix_gamma=False,
                              momentum=bn_momentum, eps=2e-5)
    relu2 = mx.symbol.Activation(data=bn2, act_type='relu', name=name_scope + "_relu2")
    conv2 = mx.symbol.Convolution(data=relu2, num_filter=n_channel, kernel=(3, 3),
                                  pad=(1, 1), stride=(1, 1), name=name_scope + "_conv2")

    sym_compute = conv2
    if stride > 1:
        sym_skip = mx.symbol.BatchNorm(data=data, fix_gamma=False, momentum=bn_momentum,
                                       eps=2e-5, name=name_scope + "_skip_bn")
        sym_skip = mx.symbol.Activation(data=sym_skip, act_type='relu', name=name_scope + "_skip_relu")
        sym_skip = mx.symbol.Convolution(data=sym_skip, num_filter=n_channel, kernel=(3, 3),
                                         pad=(1, 1), stride=(stride, stride), name=name_scope + "_skip_conv")
    else:
        sym_skip = None
    mod = sd_module.StochasticDeepModule(sym_compute, sym_skip, data_names=[name_scope + "_data"],
                                         context=context, death_rate=death_rate)

    return mod


bn_momentum = 0.9
contexts = [mx.context.gpu(i) for i in range(1)]
n_residual_blocks = 18
death_rate = 0.5
death_mode = 'linear_decay'
n_classes = 10


def get_death_rate(i_res_block):
    n_total_res_blocks = n_residual_blocks * 3
    if death_mode == 'linear_decay':
        my_death_rate = float(i_res_block) / n_total_res_blocks * death_rate
    else:
        my_death_rate = death_rate
    return my_death_rate


sym_base = mx.sym.Variable("data")
sym_base = mx.sym.Convolution(data=sym_base, num_filter=16, kernel=(3, 3),
                              pad=(1, 1), name="conv1")
sym_base = mx.sym.BatchNorm(data=sym_base, name='bn1', fix_gamma=False, momentum=bn_momentum,
                            eps=2e-5)
sym_base = mx.sym.Activation(data=sym_base, name='relu1', act_type='relu')
mod_base = mx.mod.Module(sym_base, context=contexts)

mod_seq = mx.mod.SequentialModule()
mod_seq.add(mod_base)

i_res_block = 0
for i in range(n_residual_blocks - 1):
    mod_seq.add(residual_module(get_death_rate(i_res_block), 32, 'res_B_%d' % i, contexts), auto_wiring=True)
    i_res_block += 1

mod_seq.add(residual_module(get_death_rate(i_res_block), 64, 'res_BC', context=contexts, stride=2),
            auto_wiring=True)
i_res_block += 1

for i in range(n_residual_blocks - 1):
    mod_seq.add(residual_module(get_death_rate(i_res_block), 64, 'res_C_%d' % i, context=contexts), aotu_wiring=True)
    i_res_block += 1

sym_final = mx.sym.Variable("data")
sym_final = mx.sym.Pooling(data=sym_final, kernel=(7, 7),
                           pool_type='avg', name='global_pool')
sym_final = mx.sym.FullyConnected(data=sym_final, num_hidden=n_classes, name="logits")
sym_final = mx.sym.SoftmaxOutput(data=sym_final, name="softmax")
mod_final = mx.mod.Module(sym_final, context=contexts)
mod_seq.add(mod_final, auto_wiring=True, take_labels=True)

num_examples = 60000
batch_size = 128
base_lr = .008
lr_factor = 0.5
lr_factor_epoch = 100
momentum = .9
weight_decay = 0.00001
kv_store = 'local'
initializer = mx.init.Xavier(factor_type='in', magnitude=2.34)
num_epochs = 500

epoch_size = num_examples // batch_size
lr_scheduler = mx.lr_scheduler.FactorScheduler(step=max(int(epoch_size * lr_factor_epoch), 1),
                                               factor=lr_factor)

batch_end_callbacks = [mx.callback.Speedometer(batch_size, 50)]
epoch_end_callbacks = [mx.callback.do_checkpoint("sd-%d" % (n_residual_blocks * 6 + 2))]

data_dir = os.path.join(os.path.dirname(__file__), "data", "cifar")
kv = mx.kvstore.create(kv_store)
mx.test_utils.get_cifar10()

data_shape = (3, 28, 28)
train = mx.io.ImageRecordIter(
    path_imgrec=os.path.join(data_dir, "train.rec"),
    mean_img=os.path.join(data_dir, "mean.bin"),
    data_shape=data_shape,
    batch_size=batch_size,
    rand_crop=True,
    rand_mirror=True,
    num_parts=kv.num_workers,
    part_index=kv.rank)

val = mx.io.ImageRecordIter(
    path_imgrec=os.path.join(data_dir, "test.rec"),
    mean_img=os.path.join(data_dir, "mean.bin"),
    rand_crop=False,
    rand_mirror=False,
    data_shape=data_shape,
    batch_size=batch_size,
    num_parts=kv.num_workers,
    part_index=kv.rank)

logging.basicConfig(level=logging.DEBUG)
mod_seq.fit(train, val,
            optimizer_params={"learning_rate": base_lr,
                              "momentum": momentum,
                              'lr_schedule': lr_scheduler,
                              "Wd": weight_decay},
            num_epoch=num_epochs, batch_end_callback=batch_end_callbacks,
            epoch_end_callback=epoch_end_callbacks,
            initializer=initializer)
