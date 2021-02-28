import logging
from datetime import datetime
import os
import argparse
import errno
import mxnet as mx
import numpy as np
import cv2
from scipy.io import savemat

"""
Adversarial variational auto encoder
Autoencoding beyond pixels using a
# learned similarity metric."
"""


@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, name, arr):
        arr[:] = mx.nd.array(self.value)


def encoder(nef, z_dim, batch_size, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    """

    :param nef:
    :param z_dim:
    :param batch_size:
    :param no_bias:
    :param fix_gamma:
    :param eps:
    :return: 100 dimensional embedding
    """
    '''
    The encoder is a CNN which takes 32x32 image as input
        generates the 100 dimensional shape embedding as a sample from normal distribution
        using predicted mean and variance
    '''
    BatchNorm = mx.sym.BatchNorm
    data = mx.sym.Variable("data")

    e1 = mx.sym.Convolution(data, name='enc1', kernel=(5, 5),
                            stride=(2, 2), pad=(2, 2),
                            num_filter=nef, no_bias=no_bias)
    ebn1 = BatchNorm(e1, name="enc_bn1", fix_gamma=fix_gamma, eps=eps)
    e_act1 = mx.sym.LeakyReLU(ebn1, name='enc_act1', act_type='leaky', slope=.2)

    e2 = mx.sym.Convolution(e_act1, name="enc2", kernel=(5, 5),
                            stride=(2, 2), pad=(2, 2),
                            num_filter=nef, no_bias=no_bias)
    ebn2 = BatchNorm(e2, name="enc_bn2", fix_gamma=fix_gamma, eps=eps)
    e_act2 = mx.sym.LeakyReLU(ebn2, name='enc_act2', act_type='leaky', slope=.2)

    e3 = mx.sym.Convolution(e_act2, name='enc3', kernel=(5, 5),
                            stride=(2, 2), pad=(2, 2),
                            num_filter=nef, no_bias=no_bias)
    ebn3 = BatchNorm(e3, name="enc_bn3", fix_gamma=fix_gamma, eps=eps)
    e_act3 = mx.sym.LeakyReLU(ebn3, name='enc_act3', act_type='leaky', slope=.2)

    e4 = mx.sym.Convolution(e_act3, name='enc4', kernel=(5, 5),
                            stride=(2, 2), pad=(2, 2),
                            num_filter=nef, no_bias=no_bias)
    ebn4 = BatchNorm(e4, name='enc4', fix_gamma=fix_gamma, eps=eps)
    e_act4 = mx.sym.LeakyReLU(ebn4, name='enc_act4', act_type='leaky', slope=.2)

    e_act4 = mx.sym.Flatten(e_act4)

    z_mu = mx.sym.FullyConnected(e_act4, num_hidden=z_dim, name='enc_mu')
    z_lv = mx.sym.FullyConnected(e_act4, num_hidden=z_dim, name='enc_lv')

    z = z_mu + mx.symbol.broadcast_mul(mx.symbol.exp(0.5 * z_lv),
                                       mx.symbol.random_normal(loc=0, scale=1,
                                                               shape=(batch_size, z_dim)))
    return z_mu, z_lv, z


def generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12, z_dim=100,
              activation='sigmoid'):
    """
    '''The generator is a CNN which takes 100 dimensional embedding as input
    and reconstructs the input image given to the encoder
    '''
    :param ngf:
    :param nc:
    :param no_bias:
    :param fix_gamma:
    :param eps:
    :param z_dim:
    :param activation:
    :return:
    """

    BatchNorm = mx.sym.BatchNorm

    rand = mx.sym.Variable("rand")
    rand = mx.sym.Reshape(rand, shape=(-1, z_dim, 1, 1))

    g1 = mx.sym.Deconvolution(rand, name='gen1', kernel=(5, 5),
                              stride=(2, 2), target_shape=(2, 2),
                              num_filter=ngf * 8, no_bias=no_bias)
    gbn1 = BatchNorm(g1, name='gen_bn1', fix_gamma=fix_gamma, eps=eps)
    g_act1 = mx.sym.Activation(gbn1, act_type='relu', name="gen_act1")

    g2 = mx.sym.Deconvolution(g_act1, name='gen2', kernel=(5, 5),
                              stride=(2, 2), target_shape=(4, 4),
                              num_filter=ngf * 4, no_bias=no_bias)
    gbn2 = BatchNorm(g2, name="gen_bn2", fix_gamma=fix_gamma, eps=eps)
    g_act2 = mx.sym.Activation(gbn2, name='gen_act2', act_type='relu')

    g3 = mx.sym.Deconvolution(g_act2, name='gen3', kernel=(5, 5), stride=(2, 2), target_shape=(8, 8), num_filter=ngf
                                                                                                                 * 2,
                              no_bias=no_bias)
    gbn3 = BatchNorm(g3, name='gen_bn3', fix_gamma=fix_gamma, eps=eps)
    g_act3 = mx.sym.Activation(gbn3, name='gen_act3', act_type='relu')

    g4 = mx.sym.Deconvolution(g_act3, name='gen4', kernel=(5, 5), stride=(2, 2), target_shape=(16, 16), num_filter=ngf,
                              no_bias=no_bias)
    gbn4 = BatchNorm(g4, name='gen_bn4', fix_gamma=fix_gamma, eps=eps)
    g_act4 = mx.sym.Activation(gbn4, name='gen_act4', act_type='relu')

    g5 = mx.sym.Deconvolution(g_act4, name='gen5', kernel=(5, 5), stride=(2, 2), target_shape=(32, 32), num_filter=nc,
                              no_bias=no_bias)
    gout = mx.sym.Activation(g5, name='gen_act5', act_type=activation)

    return gout


def discriminator(ndf, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    """
    '''First part of the discriminator which takes a 32x32 image as input
    and output a convolutional feature map, this is required to calculate
    the layer loss'''
    :param ndf:
    :param no_bias:
    :param fix_gamma:
    :param eps:
    :return:
    """

    BatchNorm = mx.sym.BatchNorm

    data = mx.sym.Variable('data')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(5, 5), stride=(2, 2),
                            pad=(2, 2), num_filter=ndf, no_bias=no_bias)
    d_act1 = mx.sym.LeakyReLU(d1, name='d_act1', act_type='leaky', slope=.2)

    d2 = mx.sym.Convolution(d_act1, name='d2', kernel=(5, 5), stride=(2, 2),
                            pad=(2, 2), num_filter=ndf * 2, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dis_bn2', fix_gamma=fix_gamma, eps=eps)
    d_act2 = mx.sym.LeakyReLU(dbn2, name='d_act2', act_type='leaky', slope=.2)

    d3 = mx.sym.Convolution(d_act2, name='d3', kernel=(5, 5), stride=(2, 2), pad=(2, 2), num_filter=ndf * 4,
                            no_bias=no_bias)
    dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    d_act3 = mx.sym.LeakyReLU(dbn3, name='d_act3', act_type='leaky', slope=0.2)

    return d_act3


def discriminator2(ndf, np_bias=True, fix_gamma=True, eps=1e-5 + 1e-12, ):
    """
    '''Second part of the discriminator which takes a 256x8x8 feature map as input
    and generates the loss based on whether the input image was a real one or fake one'''
    :param ndf:
    :param np_bias:
    :param fix_gamma:
    :param eps:
    :return:
    """
    BatchNorm = mx.sym.BatchNorm

    data = mx.sym.Variable("data")
    label = mx.sym.Variable("label")

    d4 = mx.sym.Convolution(data, name='d4', kernel=(5, 5), stride=(2, 2),
                            pad=(2, 2), num_filter=ndf * 8, no_bias=np_bias)
    dbn4 = BatchNorm(d4, fix_gamma=fix_gamma, name='dbn4', eps=eps)
    d_act4 = mx.sym.LeakyReLU(dbn4, name="d_act4", act_type='leaky', slope=.2)

    h = mx.sym.Flatten(d_act4)

    d5 = mx.sym.FullyConnected(h, num_hidden=1, name='d5')

    dis_loss = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dis_loss')
    return dis_loss


def gaussian_log_density(x, mu, log_var, name="GaussianLogDensity", eps=1e-6):
    """
    '''GaussianLogDensity loss calculation for layer wise loss
    '''
    :param x:
    :param mu:
    :param log_var:
    :param name:
    :param eps:
    :return:
    """
    c = mx.sym.ones_like(log_var) * 2.0 * 3.1416
    c = mx.symbol.log(c)
    var = mx.sym.exp(log_var)
    x_mu2 = mx.symbol.square(x - mu)  # [Issue] not sure the dim works or not?
    x_mu2_over_var = mx.symbol.broadcast_div(x_mu2, var + eps)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = mx.symbol.sum(log_prob, axis=1, name=name)  # keep dim=True


def discriminator_layer_loss():
    data = mx.sym.Variable("data")
    label = mx.sym.Variable("label")

    data = mx.sym.Flatten(data)
    label = mx.sym.Flatten(label)

    label = mx.sym.BlockGrad(label)

    zeros = mx.sym.zeros_like(data)

    output = -gaussian_log_density(label, data, zeros)

    dis_loss = mx.symbol.MakeLoss(mx.symbol.mean(output), name="dis_loss")

    return dis_loss


def kl_divergence_loss():
    """
    '''KLDivergenceLoss loss
    '''
    :return:
    """

    data = mx.sym.Variable("data")
    mu1, lv1 = mx.sym.split(data, num_outputs=2, axis=0)
    mu2 = mx.sym.zeros_like(mu1)
    lv2 = mx.sym.zeros_like(lv1)

    v1 = mx.sym.exp(lv1)
    v2 = mx.sym.exp(lv2)
    mu_diff_sq = mx.sym.square(mu1 - mu2)
    dim_wise_kld = .5 * ((lv2 - lv1) + mx.symbol.broadcast_div(v1, v2)
                         + mx.symbol.broadcast_div(mu_diff_sq, v2) - 1.0)
    KL = mx.symbol.sum(dim_wise_kld, axis=1)

    KL_loss = mx.symbol.MakeLoss(mx.symbol.mean(KL), name="KL_loss")

    return KL_loss


def get_data(path, activation):
    """

    :param path:
    :param activation:
    :return:
    """
    data = []
    image_names = []
    for file_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_GRAYSCALE)
        image_names.append(file_name)
        if img is not None:
            data.append(img)

    data = np.asarray(data)

    if activation == "sigmoid":
        data = data.astype(np.float32) / 255.0
    elif activation == "tanh":
        data = data.astype(np.float32) / (255.0 / 2) - 1.0
    print(data.shape)
    data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))

    np.random.seed(1200)

    p = np.random.permutation(data.shape[0])
    X = data[p]

    return X, image_names


class RandIter(mx.io.DataIter):

    def __init__(self, batch_size, n_dim):
        super(RandIter, self).__init__()
        self.batch_size = batch_size
        self.n_dim = n_dim
        self.provide_data = [('rand', (batch_size, n_dim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.n_dim, 1, 1))]


def fill_buf(buf, i, img, shape):
    m = buf.shape[0] / shape[0]

    sx = (i % m) * shape[1]
    sy = (i // m) * shape[0]
    sx = int(sx)
    sy = int(sy)
    buf[sy:sy + shape[0], sx:sx + shape[1], :] = img


def visual(title, X, activation):
    """

    :param title:
    :param X:
    :param activation:
    :return:
    """
    assert len(X.shape) == 4

    X = X.transpose(*(0, 2, 3, 1))
    if activation == "tanh":
        X = np.clip((X + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)
    elif activation == "sigmoid":
        X = np.clip(X * 255.0, 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n * X.shape[1]), int(n * X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    cv2.imwrite('%s.jpg' % title, buff)


def train(dataset, nef, ndf, ngf, nc, batch_size, Z, lr, beta1, eps, ctx, check_point,
          g_dl_weight, output_path, check_point_path, data_path, activation,
          num_epoch, save_after_every, visualize_after_every, show_after_every):
    # Encoder
    z_mu, z_lv, z = encoder(nef, Z, batch_size)
    sym_E = mx.sym.Group([z_mu, z_lv, z])

    # Generator
    sym_G = generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12,
                      z_dim=Z, activation=activation)

    # Discriminator
    h = discriminator(ndf)
    d_loss = discriminator2(ndf)
    sym_D1 = h
    sym_D2 = d_loss

    # ==============data==============
    X_train, _ = get_data(data_path, activation)
    train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size, shuffle=True)
    rand_iter = RandIter(batch_size, Z)
    label = mx.nd.zeros((batch_size,), ctx=ctx)

    # Module E
    modE = mx.mod.Module(symbol=sym_E, data_names=['data'], context=ctx)
    modE.bind(data_shapes=train_iter.provide_data)
    modE.init_params(initializer=mx.init.Normal(0.02))
    modE.init_optimizer(optimizer='adam',
                        optimizer_params={
                            'learning_rate': lr,
                            "wd": 1e-6,
                            "beta1": beta1,
                            'epsilon': eps,
                            'rescale_grad': (1.0 / batch_size)
                        })
    mods = [modE]

    # Module G
    modG = mx.mod.Module(symbol=sym_G, data_names=['rand'], context=ctx)
    modG.bind(data_shapes=rand_iter.provide_data, inputs_need_grad=True)
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(optimizer='adam',
                        optimizer_params={
                            'learning_rate': lr,
                            'wd': 1e-6,
                            'beta1': beta1,
                            'epsilon': eps
                        })

    mods.append(modG)

    # Module D
    modD1 = mx.mod.Module(sym_D1, label_names=[], context=ctx)
    modD2 = mx.mod.Module(sym_D2, label_names=['label'], context=ctx)
    modD = mx.mod.SequentialModule()
    modD.add(modD1).add(modD2, take_labels=True, auto_wiring=True)
    modD.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)

    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(optimizer='adam',
                        optimizer_params={
                            'learning_rate': lr,
                            'wd': 1e-3,
                            'beta1': beta1,
                            'epsilon': eps,
                            'rescale_grad': 1.0 / batch_size
                        })

    mods.append(modD)

    # Module DL
    symDL = discriminator_layer_loss()
    modDL = mx.mod.Module(symbol=symDL, data_names=['data'],
                          label_names=['label'], context=ctx)
    modDL.bind(data_shapes=[('data', (batch_size, nef * 4, 4, 4))],  # fix 512 here
               label_shapes=[('label', (batch_size, nef * 4, 4, 4))],
               inputs_need_grad=True)
    modDL.init_params(initializer=mx.init.Normal(0.02))
    modDL.init_optimizer(optimizer='adam',
                         optimizer_params={
                             'learning_rate': lr,
                             'wd': 0,
                             'beta1': beta1,
                             'epsilon': eps,
                             'rescale_grad': 1.0 / batch_size
                         })
    # Module KL
    symKL = kl_divergence_loss()
    modKL = mx.mod.Module(symbol=symKL, data_names=['data', ], label_names=[], context=ctx)
    modKL.bind(data_shapes=[('data', (batch_size * 2, Z))],
               inputs_need_grad=True)
    modKL.init_params(initializer=mx.init.Normal(0.02))
    modKL.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
            'epsilon': eps,
            'rescale_grad': (1.0 / batch_size)
        })
    mods.append(modKL)

    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)

    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data",
                         sort=True)
    mon = None
    if mon is not None:
        for mod in mods:
            pass

    def f_acc(label, pred):
        """
        Predict accuracy
        :param label:
        :param pred:
        :return:
        """
        pred = pred.revel()
        label = label.reval()
        return ((pred > .5) == label).mean()

    def f_entropy(label, pred):
        """
        Binary cross entropy loss
        :param label:
        :param pred:
        :return:
        """
        pred = pred.revel()
        label = label.reval()
        return -(label * np.log(pred + 1e-12) + (1.0 - label) * np.log(1 - pred + 1e-12)).mean()

    def kl_divergence(label, pred):
        """
        KL divergence loss
        :param label:
        :param pred:
        :return:
        """

        mean, log_var = np.split(pred, axis=0)
        var = np.exp(log_var)
        kl_loss = -.5 * np.sum(1 + log_var - np.power(mean, 2) - var)
        kl_loss = kl_loss / batch_size
        return kl_loss

    mG = mx.metric.CustomMetric(f_entropy)
    mD = mx.metric.CustomMetric(f_entropy)
    mE = mx.metric.CustomMetric(kl_divergence)
    mAcc = mx.metric.CustomMetric(f_acc)

    print("Training...")
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    # --------------      Train      ---------------------------
    for epoch in range(num_epoch):
        train_iter.reset()
        for t, batch in enumerate(train_iter):

            r_batch = rand_iter.next()

            if mon is not None:
                mon.tic()

            modG.forward(r_batch, is_train=True)
            outG = modG.get_outputs()

            # update discriminator on fake
            label[:] = 0
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True, )
            modD.backward()
            gradD11 = [[grad.copyto(grad.context) for grad in grads] for grads in modD1._exec_group.grad_arrays]
            gradD12 = [[grad.copyto(grad.context) for grad in grads] for grads in modD2._exec_group.grad_arrays]

            modD.update_metric(mD, [label])
            modD.update_metric(mAcc, [label])

            # Update discriminator decoded
            modE.forward(batch, is_train=True)
            mu, lv, z = modE.get_outputs()
            z = z.reshape((batch_size, Z, 1, 1))
            sample = mx.io.DataBatch([z], provide_data=[('rand', (batch_size, Z, 1, 1))])
            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()
            label[:] = 0
            modD.forward(mx.io.DataBatch(xz, [label]), is_train=True)
            modD.backward()

            # modD update
            gradD21 = [[grad.copyto(grad.context) for grad in grads] for grads in modD1._exec_group.grad_arrays]
            gradD22 = [[grad.copyto(grad.context) for grad in grads] for grads in modD2._exec_group.grad_arrays]
            modD.update_metric(mD, [label])
            modD.update_metric(mAcc, [label])

            # Update discriminator on real
            label[:] = 1
            batch.label = [label]
            modD.forward(batch, is_train=True)
            lx = [out.copyto(out.context) for out in modD1.get_outputs()]
            modD.backward()
            for grad_sr, grad_sf, grad_sd in zip(modD1._exec_group.grad_arrays, gradD11, gradD21):
                for grad_r, grad_f, grad_d in zip(grad_sr, grad_sf, grad_sd):
                    grad_r += 0.5 * (grad_f + grad_d)
            for grad_sr, grad_sf, grad_sd in zip(modD2._exec_group.grad_arrays, gradD12, gradD22):
                for grad_r, grad_f, grad_d in zip(grad_sr, grad_sf, grad_sd):
                    grad_r += 0.5 * (grad_f + grad_d)

            modD.update()
            modD.update_metric(mD, [label])
            modD.update_metric(mAcc, [label])

            modG.forward(r_batch, is_train=True)
            outG = modG.get_outputs()
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD1.get_input_grads()
            modG.backward(diffD)
            gradG1 = [[grad.copyto(grad.context) for grad in grads] for grads in modG._exec_group.grad_arrays]
            mG.update([label], modD.get_outputs())

            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()
            label[:] = 1
            modD.forward(mx.io.DataBatch(xz, [label]), is_train=True)
            modD.backward()
            diffD = modD1.get_input_grads()
            modG.backward(diffD)
            gradG2 = [[grad.copyto(grad.context) for grad in grads] for grads in modG._exec_group.grad_arrays]
            mG.update([label], modD.get_outputs())

            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()
            modD1.forward(mx.io.DataBatch(xz, []), is_train=True)
            outD1 = modD1.get_outputs()
            modDL.forward(mx.io.DataBatch(outD1, lx), is_train=True)
            modDL.backward()
            dlGrad = modDL.get_input_grads()
            modD1.backward(dlGrad)
            diffD = modD1.get_input_grads()
            modG.backward(diffD)

            for grads, gradsG1, gradsG2 in zip(modG._exec_group.grad_arrays,
                                               gradG1, gradG2):
                for grad, grad_g1, grad_g2 in zip(grads, gradG1, gradsG2):
                    grad = g_dl_weight * grad + 0.5 * (grad_g1 + grad_g2)

            modG.update()
            mG.update([label], modD.get_outputs())

            modG.forward(r_batch, is_train=True)
            outG = modG.get_outputs()
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD1.get_input_grads()
            modG.backward(diffD)
            gradG1 = [[grad.copyto(grad.context) for grad in grads] for grads in modG._exec_group.grad_arrays]
            mG.update([label], modD.get_outputs())

            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()
            label[:] = 1
            modD.forward(mx.io.DataBatch(xz, [label]), is_train=True)
            modD.backward()
            diffD = modD1.get_input_grads()
            modG.backward(diffD)
            gradG2 = [[grad.copyto(grad.context) for grad in grads] for grads in modG._exec_group.grad_arrays]
            mG.update([label], modD.get_outputs())

            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()
            modD1.forward(mx.io.DataBatch(xz, []), is_train=True)
            outD1 = modD1.get_outputs()
            modDL.forward(mx.io.DataBatch(outD1, lx), is_train=True)
            modDL.backward()
            dlGrad = modDL.get_input_grads()
            modD1.backward(dlGrad)
            diffD = modD1.get_input_grads()
            modG.backward(diffD)

            for grads, gradsG1, gradsG2 in zip(modG._exec_group.grad_arrays, gradG1, gradG2):
                for grad, grad_g1, grad_g2 in zip(grads, gradsG1, gradsG2):
                    grad = g_dl_weight * grad + 0.5 * (grad_g1 + grad_g2)

            modG.update()
            mG.update([label], modD.get_outputs())

            modG.forward(sample, is_train=True)
            xz = modG.get_outputs()

            # Generator update
            modD1.forward(mx.io.DataBatch(xz, []), is_train=True)
            outD1 = modD1.get_outputs()
            modDL.forward(mx.io.DataBatch(outD1, lx), is_train=True)
            DL_loss = modDL.get_outputs()
            modDL.backward()
            dlGrad = modDL.get_input_grads()
            modD1.backward(dlGrad)
            diffD = modD1.get_input_grads()
            modG.backward(diffD)

            # Update Encoder
            nElements = batch_size
            modKL.forward(mx.io.DataBatch([mx.ndarray.concat(mu, lv, dim=0)]), is_train=True)
            KL_loss = modKL.get_outputs()
            modKL.backward()
            gradKLLoss = modKL.get_input_grads()
            diffG = modG.get_input_grads()
            diffG = diffG[0].reshape((batch_size, Z))
            modE.backward(mx.ndarray.split(gradKLLoss[0], num_outputs=2, axis=0) + [diffG])
            modE.update()
            pred = mx.ndarray.concat(mu, lv, dim=0)
            mE.update([pred], [pred])
            if mon is not None:
                mon.toc_print()

            t += 1
            if t % show_after_every == 0:
                print('epoch: ', epoch, 'iter: ', t, 'metric: ', mAcc.get(),
                      mG.get(), mD.get(), mE.get(), KL_loss[0].asnumpy(), DL_loss[0].asnumpy())
                mAcc.reset()
                mG.reset()
                mD.reset()
                mE.reset()

            if epoch % visualize_after_every == 0:
                visual(output_path + "g_out" + str(epoch), outG[0].asnumpy(), activation)
                visual(output_path + 'data' + str(epoch), batch.data[0].asnumpy(), activation)

            if check_point and epoch % save_after_every == 0:
                print("Saving to file...")
                modG.save_params(check_point_path + '%s_G-%04d.params' % (dataset, epoch))
                modD.save_params(check_point_path + '%s_D-04%d.params' % (dataset, epoch))
                modE.save_params(check_point_path + '%s_D-04%d.params' % (dataset, epoch))


def test(nef, ngf, nc, batch_size, Z, ctx, pre_trained_encoder_path,
         pre_train_generator_path, output_path, data_path, activation,
         save_embedding, embedding_path):
    z_mu, z_lv, z = encoder(nef, Z, batch_size)
    symE = mx.sym.Group([z_mu, z_lv, z])

    # Generator
    symG = generator(ngf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12,
                     z_dim=Z, activation=activation)
    # ---------------Data-------------------
    X_test, image_names = get_data(data_path, activation)
    test_iter = mx.io.NDArrayIter(X_test, batch_size=batch_size, shuffle=False)

    # Module E
    modE = mx.mod.Module(symbol=symE, data_names=['data'], context=ctx)
    modE.bind(data_shapes=test_iter.provide_data)
    modE.load_params(pre_trained_encoder_path)

    # Module G
    modG = mx.mod.Module(symbol=symG, data_names=['rand'],
                         context=ctx)
    modG.bind(data_shapes=[('rand', (1, Z, 1, 1))])
    modG.load_params(pre_train_generator_path)

    print("Testing....")

    # Test
    test_iter.reset()
    for t, batch in enumerate(test_iter):
        modE.forward(batch, is_train=False)
        mu, lv, z = modE.get_outputs()
        mu = mu.reshape((batch_size, Z, 1, 1))
        sample = mx.io.DataBatch([mu], label=None, provide_data=[('rand', (batch_size, Z, 1, 1))])
        modG.forward(sample, is_train=False)
        outG = modG.get_outputs()

        visual(output_path + '/' + 'gout' + str(t), outG[0].asnumpy(), activation)
        visual(output_path + '/' + 'data' + str(t), batch.data[0].asnumpy(), activation)
        image_name = image_names[t].split('.')[0]

        if save_embedding:
            savemat(embedding_path + '/' + image_name + '.mat', {'embedding': mu.asnumpy()})


def create_and_validate_dir(data_dir):
    if data_dir != "":
        if not os.path.exists(data_dir):
            try:
                logging.info("Create directory %s", data_dir)
                os.makedirs(data_dir)
            except OSError as exec:
                if exec.errno != errno.EEXIST:
                    raise OSError("failed to create " + data_dir)


def parse_args():
    '''Parse args
        '''
    parser = argparse.ArgumentParser(description='Train and Test an Adversarial Variatiional Encoder')

    parser.add_argument('--train', help='train the network', default=True, action='store_true')
    parser.add_argument('--test', help='test the network', action='store_true')
    parser.add_argument('--save_embedding', help='saves the shape embedding of each input image', action='store_true')
    parser.add_argument('--dataset', help='dataset name', default='caltech', type=str)
    parser.add_argument('--activation', help='activation i.e. sigmoid or tanh', default='sigmoid', type=str)
    parser.add_argument('--training_data_path', help='training data path',
                        default='E:\Python_Data\caltech101', type=str)
    parser.add_argument('--testing_data_path', help='testing data path', default='E:\Python_Data\caltech101',
                        type=str)
    parser.add_argument('--pretrained_encoder_path', help='pretrained encoder model path',
                        default='checkpoints32x32_sigmoid/caltech_E-0045.params', type=str)
    parser.add_argument('--pretrained_generator_path', help='pretrained generator model path',
                        default='checkpoints32x32_sigmoid/caltech_G-0045.params', type=str)
    parser.add_argument('--output_path', help='output path for the generated images', default='outputs32x32_sigmoid',
                        type=str)
    parser.add_argument('--embedding_path', help='output path for the generated embeddings',
                        default='outputs32x32_sigmoid', type=str)
    parser.add_argument('--checkpoint_path', help='checkpoint saving path ', default='checkpoints32x32_sigmoid',
                        type=str)
    parser.add_argument('--nef', help='encoder filter count in the first layer', default=64, type=int)
    parser.add_argument('--ndf', help='discriminator filter count in the first layer', default=64, type=int)
    parser.add_argument('--ngf', help='generator filter count in the second last layer', default=64, type=int)
    parser.add_argument('--nc',
                        help='generator filter count in the last layer i.e. 1 for grayscale image, 3 for RGB image',
                        default=1, type=int)
    parser.add_argument('--batch_size', help='batch size, keep it 1 during testing', default=64, type=int)
    parser.add_argument('--Z', help='embedding size', default=100, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.0002, type=float)
    parser.add_argument('--beta1', help='beta1 for adam optimizer', default=0.5, type=float)
    parser.add_argument('--epsilon', help='epsilon for adam optimizer', default=1e-5, type=float)
    parser.add_argument('--g_dl_weight', help='discriminator layer loss weight', default=1e-1, type=float)
    parser.add_argument('--gpu', help='gpu index', default=0, type=int)
    parser.add_argument('--use_cpu', help='use cpu', action='store_true')
    parser.add_argument('--num_epoch', help='number of maximum epochs ', default=45, type=int)
    parser.add_argument('--save_after_every', help='save checkpoint after every this number of epochs ', default=5,
                        type=int)
    parser.add_argument('--visualize_after_every', help='save output images after every this number of epochs',
                        default=5, type=int)
    parser.add_argument('--show_after_every', help='show metrics after this number of iterations', default=10, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.test and not os.path.exists(args.testing_data_path):
        if not os.path.exists(args.testing_data_path):
            raise OSError("Provided Testing Path: {} does not exits".format(args.testing_data_path))
        if not os.path.exists(args.checkpoint_path):
            raise OSError("Provided checkpoint path: {} does not exist".format(args.checkpoint_path))
    create_and_validate_dir(args.checkpoint_path)
    create_and_validate_dir(args.output_path)

    if args.use_cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    checkpoint = True

    if args.train:
        print("Train")
        train(args.dataset, args.nef, args.ndf, args.ngf, args.nc,
              args.batch_size, args.Z, args.lr, args.beta1, args.epsilon,
              ctx, checkpoint, args.g_dl_weight, args.output_path,
              args.checkpoint_path, args.training_data_path,
              args.activation, args.num_epoch, args.save_after_every,
              args.visualize_after_every, args.show_after_every)
    if args.test:
        test(args.nef, args.ngf, args.nc, 1, args.Z, ctx,
             args.pretrained_encoder_path, args.pretrained_generator_path,
             args.output_path, args.testing_data_path, args.activation,
             args.save_embedding, args.embedding_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()


