import mxnet as mx
import numpy as np
from mxnet import autograd


class Sqr(mx.operator.CustomOp):
    """
    Square NDArray
    """

    def forward(self, is_train, req, in_data, out_data, aux):
        inp = in_data[0]
        if inp.stype == 'csr':
            csr_m = inp.data
            out = mx.nd.sparse.csr_matrix((csr_m * csr_m, inp.indices, inp.indptr), shape=inp.shape)
        else:
            out = inp * inp
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 2 * mx.nd.sparse.elemwise_mul(in_data[0], out_grad[0]))


@mx.operator.register("sqr")
class SqrProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SqrProp, self).__init__(need_top_grad=True)

    def list_outputs(self):
        return ['output']

    def list_arguments(self):
        return ['data']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def infer_storage_type(self, in_stype):
        """
            Infer storage type logic for the forward pass
            Takes a list of storage types for inputs
            Returns three lists lists, one for input storage types inferred,
            second for output storage types inferred and third for aux storage
            types inferred
            The in_stype is the list containing storage type for inputs
            If the input is a dense ndarray then we infer the input
            and output to be dense. If input is csr then input and output
            are inferred as csr.
        """
        if in_stype[0] == 'default':
            return ['default'], ['default'], []
        return ['csr'], ['csr'], []

    def infer_storage_type_backward(self, ograd_stype, in_stype, out_stype, igrad_stype, aux_stype):
        """
            Infer storage type logic for the backward pass
            Takes storage type of output gradients(ograd_stype), inputs(in_stype),
            outputs(out_stype) and aux(aux_stype).
            Returns inferred storage types in the following order:
            ograd_stype, in_stype, out_stype, igrad_stype (Storage type for input gradients)
            and aux_stype.
        """
        if in_stype[0] == 'default':
            return ['default'], ['default'], ['default'], ['default'], []
        return ['csr'], ['csr'], ['csr'], ['default'], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Sqr()


x = mx.nd.array(np.random.uniform(1, 10, size=(4, 10)))
x.attach_grad()
z = mx.nd.ones((4, 10))

with autograd.record():
    y = mx.nd.Custom(x, op_type='sqr')
    y.backward(out_grad=z)
print("Original ndarray")
print("--------------")
print(x.asnumpy())
print("Squared ndarray")
print("--------------")
print(y.asnumpy())
print("stype of input is {}".format(x.stype))
print("stype of output is {}".format(y.stype))
print("Grad ndarray")
print(x.grad.asnumpy())

x = mx.nd.array(np.random.uniform(1, 10, size=(4, 10)))
x = x.tostype('csr')
x.attach_grad(stype='default')
z = mx.nd.ones((4, 10))
z = z.tostype('csr')
with autograd.record():
    y = mx.nd.Custom(x, op_type='sqr')
    y.backward(out_grad=z)
print("Original ndarray")
print("--------------")
print(x.asnumpy())
print("Squared ndarray")
print("--------------")
print(y.asnumpy())
print("stype of input is {}".format(x.stype))
print("stype of output is {}".format(y.stype))
print("Grad ndarray")
print(x.grad.asnumpy())