import mxnet as mx


def squash(data, squash_axis, name=''):
    epsilon = 1e-08
    s_square_norm = mx.sym.sum(data=mx.sym.square(data, name='square_' + name),
                               axis=squash_axis, keepdims=True, name='s_squared_norm_' + name)
    scale = s_square_norm / (1 + s_square_norm) / mx.sym.sqrt(data=(s_square_norm + epsilon),
                                                              name='s_squared_norm_sqrt_' + name)
    squash_net = mx.sym.broadcast_mul(scale, data, name='squashed_net_' + name)
    return squash_net


def primary_caps(data, dim_vector, n_channels, kernel, strides, name=''):
    out = mx.sym.Convolution(data=data,
                             num_filter=dim_vector * n_channels,
                             kernel=kernel,
                             stride=strides,
                             name=name)
    out = mx.sym.Reshape(out, shape=(0, -1, dim_vector))
    out = squash(out, squash_axis=2)
    return out


class CapsuleLayer:
    def __init__(self, num_capsule, dim_vector, batch_size, kernel_initializer,
                 bias_initializer, num_routing=3):
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.batch_size = batch_size
        self.num_routing = num_routing
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def __call__(self, data, *args, **kwargs):
        _, out_shapes, _ = data.infer_shape(data=(self.batch_size, 1, 28, 28))
        _, input_num_capsule, input_dim_vector = out_shapes[0]

        # build w and bias
        # W : (input_num_capsule, num_capsule, input_dim_vector, dim_vector)
        # bias : (batch_size, input_num_capsule, num_capsule ,1, 1)
        w = mx.sym.Variable("weight",
                            shape=(1, input_num_capsule, self.num_capsule,
                                   input_dim_vector, self.dim_vector),
                            init=self.kernel_initializer)
        bias = mx.sym.Variable("bias",
                               shape=(self.batch_size, input_num_capsule,
                                      self.num_capsule, 1, 1),
                               init=self.bias_initializer)
        bias = mx.sym.BlockGrad(bias)
        bias_ = bias

        inputs_expand = mx.sym.Reshape(data=data, shape=(0, 0, -4, -1, 1))
        inputs_expand = mx.sym.Reshape(data=inputs_expand, shape=(0, 0, -4, 1, -1, 0))
        inputs_tiled = mx.sym.tile(data=inputs_expand, reps=(1, 1, self.num_capsule, 1, 1))
        w_tiled = mx.sym.tile(data=w, reps=(self.batch_size, 1, 1, 1, 1))

        inputs_hat = mx.sym.linalg_gemm2(w_tiled, inputs_tiled, transpose_a=True)

        inputs_hat = mx.sym.swapaxes(data=inputs_hat, dim1=3, dim2=4)
        inputs_hat_stopped = inputs_hat
        inputs_hat_stopped = mx.sym.BlockGrad(inputs_hat_stopped)

        for i in range(0, self.num_routing):
            c = mx.sym.softmax(data=bias_, axis=2, name='c' + str(i))
            if i == self.num_routing - 1:
                outputs = squash(mx.sym.sum(mx.sym.broadcast_mul(c, inputs_hat), axis=1, keepdims=True),
                                 squash_axis=4)
            else:
                outputs = squash(mx.sym.sum(mx.sym.broadcast_mul(c, inputs_hat_stopped, name='broadcast_mul_' + str(
                    i)), axis=1, keepdims=True, name='sum_' + str(i)), name='output_' + str(i), squash_axis=4)
                bias_ = bias_ + mx.sym.sum(mx.sym.broadcast_mul(c, inputs_hat_stopped,
                                                                name='bias_broadcast_mul_' + str(i)), axis=4,
                                           keepdims=True, name='bias_' + str(i))
        outputs = mx.sym.Reshape(data=outputs, shape=(-1, self.num_capsule, self.dim_vector))
        return outputs
