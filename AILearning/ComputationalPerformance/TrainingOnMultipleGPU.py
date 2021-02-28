from d2l import AllDeepLearning as d2l
from mxnet import nd, autograd, gluon, gpu, cpu
from mxnet.gluon import nn
from mxnet import nd as npx
from mxnet import nd as np

scale = 0.01
W1 = nd.random.normal(scale=scale, shape=(20, 1, 3, 3))
b1 = nd.zeros(20)
W2 = nd.random.normal(scale=scale, shape=(50, 20, 5, 5))
b2 = nd.zeros(50)
W3 = nd.random.normal(scale=scale, shape=(800, 128))
b3 = nd.zeros(128)
W4 = nd.random.normal(scale=scale, shape=(128, 10))
b4 = nd.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]


# Cross-entropy loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()
def lenet(X, params):
    h1_conv = nd.Convolution(data=X, weight=params[0], bias=params[1], kernel=(3, 3),
                             num_filter=20)
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='avg', kernel=(2, 2), stride=(2, 2))

    h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3], kernel=(5, 5),
                             num_filter=50)
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(h2_activation, pool_type='avg', kernel=(2, 2), stride=(2, 2))

    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = nd.dot(h2, params[4] + params[5])
    h3 = nd.relu(h3_linear)
    y_hat = nd.dot(h3, params[6]) + params[7]
    return y_hat


def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params


def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])


def split_batch(X, y, ctx_list):
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, ctx_list)), \
        gluon.utils.split_and_load(y, ctx_list)


# def train_batch(X, y, gpu_params, ctx_list, lr):
#     gpu_Xs, gpu_ys = split_batch(X, y, ctx_list)
#     with autograd.record():  # Loss is calculated separately on each GPU
#         losses = [loss(lenet(gpu_X, gpu_W), gpu_y)
#                   for gpu_X, gpu_y, gpu_W in zip(gpu_Xs, gpu_ys, gpu_params)]
#     for l in losses:  # Back Propagation is performed separately on each GPU
#         l.backward()
#     # Sum all gradients from each GPU and broadcast them to all GPUs
#     for i in range(len(gpu_params[0])):
#         allreduce([gpu_params[c][i].grad for c in range(len(ctx_list))])
#     # The model parameters are updated separately on each GPU
#     for param in gpu_params:
#         d2l.sgd(param, lr, X.shape[0])  # H
def train_batch(X, y, gpu_params, ctx_list, lr):
    gpu_Xs, gpu_ys = split_batch(X, y, ctx_list)
    with autograd.record():
        losses = [loss(lenet(gpu_X, gpu_W), gpu_y)
                  for gpu_X, gpu_y, gpu_W in zip(gpu_Xs, gpu_ys, gpu_params)]
    for l in losses:
        l.backward()

    for i in range(len(gpu_params[0])):
        allreduce([gpu_params[c][i].grad for c in range(len(ctx_list))])

    for param in gpu_params:
        d2l.sgd(param, lr, X.shape[0])


def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx_list = [d2l.try_gpu(i) for i in range(num_gpus)]

    gpu_prams = [get_params(params, c) for c in ctx_list]

    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            train_batch(X, y, gpu_prams, ctx_list, lr)
            nd.waitall()
        timer.stop()

        animator.add(epoch+1, (d2l.evaluate_accuracy_gpu(lambda x: lenet(x, gpu_prams[0]),
                                                         test_iter, ctx[0]),))
    print('test acc: %.2f, %.1f sec/epoch on %s' % (animator.Y[0][-1], timer.avg(), ctx_list))


# new_params = get_params(params, d2l.try_gpu(0))

# data = nd.arange(20).reshape(4, 5)
ctx = [gpu(0), cpu()]
# split = gluon.utils.split_and_load(data, ctx)
train(num_gpus=1, batch_size=256, lr=0.2)
d2l.plt.show()