import logging
import numpy as np
import mxnet as mx
from AI.MxnetExamples.DeepEmbedded.Model import MXModel
import AI.MxnetExamples.DeepEmbedded.Model as model
from AI.MxnetExamples.DeepEmbedded.Solver import Solver, Monitor


class AutoEncoderModel(MXModel):
    def setup(self, dims, sparseness_penalty=None, pt_dropout=None,
              ft_dropout=None, input_act=None, internal_act="relu",
              output_act=None, *args, **kwargs):
        self.N = len(dims) - 1
        self.dims = dims
        self.stacks = []
        self.pt_dropout = pt_dropout
        self.ft_dropout = ft_dropout
        self.input_act = input_act
        self.internal_act = input_act
        self.output_act = output_act

        self.data = mx.symbol.Variable("data")
        for i in range(self.N):
            if i == 0:
                decoder_act = input_act
                i_dropout = None
            else:
                decoder_act = internal_act
                i_dropout = pt_dropout
            if i == self.N - 1:
                encoder_act = output_act
                o_dropout = None
            else:
                encoder_act = internal_act
                o_dropout = pt_dropout
            i_stack, i_args, i_args_grad, i_args_mult, i_auxs = self.make_stack(
                i, self.data, dims[i], dims[i + 1], sparseness_penalty,
                i_dropout, o_dropout, encoder_act, decoder_act)

            self.stacks.append(i_stack)
            self.args.update(i_args)
            self.args_grad.update(i_args_grad)
            self.args_mult.update(i_args_mult)
            self.auxs.update(i_auxs)
        self.encoder, self.internals = self.make_encoder(self.data, dims, sparseness_penalty,
                                                         ft_dropout, input_act, output_act)
        self.decoder = self.make_decoder(
            self.encoder, dims, sparseness_penalty, ft_dropout, input_act, output_act)

        if input_act == 'softmax':
            self.loss = self.decoder
        else:
            self.loss = mx.symbol.LinearRegressionOutput(data=self.decoder,
                                                         label=self.data)

    def make_stack(self, i_stack, data, num_input, num_hidden, sparseness_penalty=None,
                   i_dropout=None, o_dropout=None, encoder_act='relu', decoder_act='relu'):
        x = data
        if i_dropout:
            x = mx.symbol.Dropout(data=x, p=i_dropout)
        x = mx.symbol.FullyConnected(name='encoder_%d' % i_stack,
                                     data=x, num_hidden=num_hidden)
        if encoder_act:
            x = mx.symbol.Activation(data=x, act_type=encoder_act)
            if encoder_act == 'sigmoid' and sparseness_penalty:
                x = mx.symbol.IdentityAttachKLSparseReg(
                    data=x, name='sparse_encoder_%d' % i_stack,
                    num_hidden=num_input)
        if o_dropout:
            x = mx.symbol.Dropout(data=x, p=o_dropout)
        x = mx.symbol.FullyConnected(name='decoder_%d' % i_stack,
                                     data=x, num_hidden=num_input)
        if decoder_act:
            x = mx.symbol.Activation(data=x, act_type=decoder_act)
            if decoder_act == 'sigmoid' and sparseness_penalty:
                x = mx.symbol.IdentityAttachKLSparseReg(
                    data=x, name='sparse_decoder_%d' % i_stack,
                    penalty=sparseness_penalty)
            x = mx.symbol.LinearRegressionOutput(data=x, label=data)
        else:
            x = mx.symbol.LinearRegressionOutput(data=x, label=data)

        args = {'encoder_%d_weight' % i_stack: mx.nd.empty((num_hidden, num_input), self.xpu),
                'encoder_%d_bias' % i_stack: mx.nd.empty((num_hidden,), self.xpu),
                'decoder_%d_weight' % i_stack: mx.nd.empty((num_input, num_hidden), self.xpu),
                'decoder_%d_bias' % i_stack: mx.nd.empty((num_input,), self.xpu), }
        args_grad = {'encoder_%d_weight' % i_stack: mx.nd.empty((num_hidden, num_input), self.xpu),
                     'encoder_%d_bias' % i_stack: mx.nd.empty((num_hidden,), self.xpu),
                     'decoder_%d_weight' % i_stack: mx.nd.empty((num_input, num_hidden), self.xpu),
                     'decoder_%d_bias' % i_stack: mx.nd.empty((num_input,), self.xpu), }
        args_mult = {'encoder_%d_weight' % i_stack: 1.0,
                     'encoder_%d_bias' % i_stack: 2.0,
                     'decoder_%d_weight' % i_stack: 1.0,
                     'decoder_%d_bias' % i_stack: 2.0, }

        auxs = {}
        if encoder_act == 'sigmoid' and sparseness_penalty:
            auxs['sparse_encoder_%d_moving_avg' % i_stack] = \
                mx.nd.ones(num_hidden, self.xpu) * .5
        if decoder_act == 'sigmoid' and sparseness_penalty:
            auxs['sparse_decoder_%d_moving_avg' % i_stack] = \
                mx.nd.ones(num_input, self.xpu) * 0.5
        init = mx.initializer.Uniform(.07)
        for k, v in args.items():
            init(mx.initializer.InitDesc(k), v)

        return x, args, args_grad, args_mult, auxs

    @staticmethod
    def make_encoder(data, dims, sparseness_penalty=None, dropout=None,
                     internal_act='relu', output_act=None):
        x = data
        internals = []
        N = len(dims) - 1
        for i in range(N):
            x = mx.symbol.FullyConnected(name='encoder_%d' % i, data=x,
                                         num_hidden=dims[i + 1])
            if internal_act and i < N - 1:
                x = mx.symbol.Activation(data=x, act_type=internal_act)
                if internal_act == 'sigmoid' and sparseness_penalty:
                    x = mx.symbol.IdentityAttachKLSparseReg(
                        data=x, name='sparse_encoder_%d' % i, penalty=sparseness_penalty)
            elif output_act and i == N - 1:
                x = mx.symbol.Activation(data=x, act_type=output_act)
                if output_act == 'sigmoid' and sparseness_penalty:
                    x = mx.symbol.IdentityAttachKLSparseReg(
                        data=x, name='sparse_encoder_%d' % i, penalty=sparseness_penalty)
            if dropout:
                x = mx.symbol.Dropout(data=x, p=dropout)
            internals.append(x)
        return x, internals

    @staticmethod
    def make_decoder(feature, dims, sparseness_penalty=None,
                     dropout=None, internal_act='relu', input_act=None):
        x = feature
        N = len(dims) - 1
        for i in reversed(range(N)):
            x = mx.symbol.FullyConnected(name='decoder_%d' % i,
                                         data=x, num_hidden=dims[i])
            if internal_act and i > 0:
                x = mx.symbol.Activation(data=x, act_type=internal_act)
                if internal_act == 'sigmoid' and sparseness_penalty:
                    x = mx.symbol.IdentityAttachKLSparseReg(
                        data=x, name='sparse_decoder_%d' % i, penalty=sparseness_penalty)
                elif internal_act and i == 0:
                    x = mx.symbol.Activation(data=x, act_type=internal_act)
                    if input_act == 'sigmoid' and sparseness_penalty:
                        x = mx.symbol.IdentityAttachKLSparseReg(
                            data=x, name='sparse_decoder_%d' % i, penalty=sparseness_penalty)
                if dropout and i > 0:
                    x = mx.symbol.Dropout(data=x, p=dropout)
            return x

    def layer_wise_pretrain(self, X, batch_size, n_iter, optimizer, lr, decay,
                            lr_scheduler=None, print_every=1000):
        def l2_norm(label, pred):
            return np.mean(np.square(label - pred)) / 2.0

        solver = Solver(optimizer, momentum=.9, wd=decay,
                        learning_rate=lr, lr_scheduler=lr_scheduler)
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(print_every))
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size,
                                      shuffle=True, last_batch_handle='roll_over')

        for i in range(self.N):
            if i == 0:
                data_iter_i = data_iter
            else:
                X_i = list(model.extract_feature(
                    self.internals[i - 1], self.args, self.auxs, data_iter,
                    X.shape[0], self.xpu
                ).values())[0]
                data_iter_i = mx.io.NDArrayIter({'data': X_i}, batch_size=batch_size,
                                                last_batch_handle='roll_over')
            logging.info("Pre training layer %d..." % i)
            solver.solve(self.xpu, self.stacks[i],
                         self.args, self.args_grad, self.auxs,
                         data_iter_i, 0, n_iter, {}, False)

    def fine_tune(self, X, batch_size, n_iter, optimizer, lr, decay, lr_scheduler=None,
                  print_every=1000):
        def l2_norm(label, pred):
            return np.mean(np.square(label - pred)) / 2.0

        solver = Solver(optimizer, momentum=0.9, wd=decay, learning_rate=lr,
                        lr_scheduler=lr_scheduler)
        solver.set_metric(mx.metric.CustomMetric(l2_norm))
        solver.set_monitor(Monitor(print_every))
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size,
                                      shuffle=True, last_batch_handle='roll_over')
        logging.info("Fine tuning...")
        solver.solve(self.xpu, self.loss, self.args, self.args_grad, self.auxs,
                     data_iter, 0, n_iter, {}, False)

    def eval(self, X):
        batch_size = 100
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size,
                                      shuffle=False, last_batch_handle='pad')
        Y = list(model.extract_feature(self.loss, self.args, self.auxs,
                                       data_iter, X.shape[0], self.xpu).values())[0]
        return np.mean(np.square(Y - X)) / 2.0
