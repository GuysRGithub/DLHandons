import logging
import numpy as np
import mxnet as mx


class Softmax(mx.operator.CustomOp):
    def __init__(self):
        super(Softmax, self).__init__()
        forward_src = r"""
            template<class DType>
            __global__ void fwd(const DType* x, DType* y, const int row_size, const int req) {
                const int offset = row_size * threadIdx.x;
                DType max = x[offset];
                for(int i = 1; i < row_size; ++i) {
                    if(max < x[offset + i]) {
                        max = x[offset + i];
                    }
                }
                DType sum = 0;
                for(int i = 0; i < row_size; ++i) {
                    sum += exp(x[offset + i] - max);
                }
                switch(req) {
                    case 1:
                        for(int i = 0; i < row_size; ++i) {
                            y[offset + i] = exp(x[offset + i] - max) / sum;
                        }
                        break;
                    case 2:
                        for(int i = 0; i < row_size; ++i) {
                            y[offset + i] += exp(x[offset + i] - max) / sum;
                        }
                        break;
                }
            }
        """
        backward_src = r"""
            template<class DType>
            __global__ void bwd(const DType* l, const DType* y, DType* dx, const int req) {
                const int z = static_cast<int>(l[blockIdx.x]);
                const int i = threadIdx.x + blockDim.x * blockIdx.x;
                if(req == 1) {
                    dx[i]  = threadIdx.x == z ? y[i] - 1 : y[i];
                } else {
                    dx[i] += threadIdx.x == z ? y[i] - 1 : y[i];
                }
            }
        """
        forward_kernel_mod = mx.rtc.CudaModule(forward_src,
                                               exports=("fwd<float>", "fwd<double>"))
        backward_kernel_mod = mx.rtc.CudaModule(backward_src,
                                                exports=("bwd<float>", "bwd<double>"))

        forward_kernel_float_signature = "const float*, const float*, const int, const int"
        self.forward_float_kernel = forward_kernel_mod.get_kernel("fwd<float>",
                                                                  forward_kernel_float_signature)

        backward_kernel_float_signature = "const float*, const float*, float*, const int"
        self.backward_float_kernel = backward_kernel_mod.get_kernel("bwd<float>",
                                                                    backward_kernel_float_signature)

        forward_kernel_double_signature = "const double*, const double*, const int, const int"
        self.forward_double_kernel = forward_kernel_mod.get_kernel("fwd<double>",
                                                                   forward_kernel_double_signature)

        backward_kernel_double_signature = "const double*, const double*, double*, const int"
        self.backward_double_kernel = backward_kernel_mod.get_kernel("bwd<double>",
                                                                     backward_kernel_double_signature)

    def forward(self, is_train, req, in_data, out_data, aux):
        if req[0] == "null":
            return
        x = in_data[0]
        y = out_data[0]

        if y.dtype == np.float64:
            self.forward_double_kernel.launch((x, y, x.shape[1], self._req_code(req[0])),
                                              mx.gpu(0), (1, 1, 1),
                                              (x.shape[0], 1, 1))
        else:
            self.forward_float_kernel.launch((x, y, x.shape[1], self._req_code(req[0])),
                                             mx.gpu(0), (1, 1, 1),
                                             (x.shape[0], 1, 1))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if req[0] == "null":
            return
        label = in_data[1]
        y = out_data[0]  # output from forward
        dx = in_grad[0]  # gradient

        if dx.dtype == np.float64:
            self.backward_double_kernel.launch((label, y, dx, self._req_code(req[0])),
                                               mx.gpu(0),
                                               (y.shape[0], 1, 1),
                                               (y.shape[1], 1, 1))
        else:
            self.backward_float_kernel.launch((label, y, dx, self._req_code(req[0])),
                                              mx.gpu(0), (y.shape[0], 1, 1),
                                              (y.shape[1], 1, 1))

    @staticmethod
    def _req_code(req):
        if req == "write":
            return 1
        elif req == "add":
            return 2
        elif req == "null":
            return 0
        else:
            raise ValueError("Invalid value of `red`: {}".format(req))


@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ["data", "label"]

    def list_outputs(self):
        return ["output"]

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

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

train, val = mx.test_utils.get_mnist_iterator(batch_size=100, input_shape=(784,))

context = mx.gpu(0)

mod = mx.mod.Module(mlp, context=context)

mod.fit(train_data=train, eval_data=val, optimizer='sgd',
        optimizer_params={'learning_rate': .1, 'momentum': .9, 'wd': 0.00001},
        num_epoch=12, batch_end_callback=mx.callback.Speedometer(100, 100))
