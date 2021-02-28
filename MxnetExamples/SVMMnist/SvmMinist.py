import logging
import random

import mxnet as mx
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

np.random.seed(1234)
mx.random.seed(1234)
random.seed(1234)

data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=512)
act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')
fc2 = mx.symbol.FullyConnected(data=fc1, name='fc2', num_hidden=512)
act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type='relu')
fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)

mlp_svm_l2 = mx.symbol.SVMOutput(data=fc3, name='svm_l2')
mlp_svm_l1 = mx.symbol.SVMOutput(data=fc3, name="svm_l1", use_linear=True)
mlp_softmax = mx.symbol.SoftmaxOutput(data=fc3, name="softmax")

print("Preparing Data")
mnist_data = mx.test_utils.get_mnist()
X = np.concatenate([mnist_data["train_data"], mnist_data["test_data"]])
Y = np.concatenate([mnist_data["train_label"], mnist_data["test_label"]])
X = X.reshape((X.shape[0], -1)).astype(np.float32) * 255

mnist_pca = PCA(n_components=70).fit_transform(X)
noise = np.random.normal(size=mnist_pca.shape)
mnist_pca += noise
p = np.random.permutation(mnist_pca.shape[0])
X = mnist_pca[p] / 255
Y = Y[p]
X_show = X[p]

# This is just to normalize the input and separate train set and test set
X_train = X[:60000]
X_test = X[60000:]
X_show = X_show[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]

batch_size = 200

ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

results = {}

for output in [mlp_svm_l2, mlp_svm_l1, mlp_softmax]:
    print("\nTesting with %s\n" % output.name)
    label = output.name + "_label"

    train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size, label_name=label)
    test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size, label_name=label)

    mod = mx.mod.Module(context=ctx, symbol=output, label_names=[label])
    mod.fit(train_data=train_iter, eval_data=test_iter,
            batch_end_callback=mx.callback.Speedometer(batch_size, 200),
            num_epoch=10,
            optimizer_params={
                'learning_rate': 0.1,
                "momentum": 0.9,
                "wd":  0.00001,
            })
    results[output.name] = mod.score(test_iter, mx.metric.Accuracy())[0][1] * 100
    print("Accuracy for %s: " % output.name, mod.score(test_iter, mx.metric.Accuracy())[0][1]*100, '%\n')

for key, value in results.items():
    print(key, value, "%s")
