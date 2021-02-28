import random

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd

# Train 128 sample every epoch
batch_size = 128
epochs = 5
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
lr = 0.01

# (60000, 28, 28, 1): (sample, width, height, batch)
train_dataset = gluon.data.vision.MNIST(train=True)

# (10000, 28, 28, 1): (sample, width, height, batch)
test_dataset = gluon.data.vision.MNIST(train=False)


def transform(x, y):
    # transpose to take batch_size to 0 for mxnet usage
    x = x.transpose((2, 0, 1)).astype('float32') / 255.0
    y1 = y
    y2 = y % 2  # Odd or Even (1 or 0)
    return x, np.float32(y1), np.float32(y2)


train_dataset_t = train_dataset.transform(transform)
test_dataset_t = test_dataset.transform(transform)

train_data = gluon.data.DataLoader(train_dataset_t, batch_size,
                                   shuffle=True, last_batch='rollover',)
test_data = gluon.data.DataLoader(test_dataset_t, shuffle=False,
                                  last_batch='rollover', batch_size=batch_size,)

print("Input shape: {}, Target Labels: {}".format(train_dataset[0][0].shape, train_dataset_t[0][1:]))


class MultiTaskNetwork(gluon.HybridBlock):

    def __init__(self):
        super(MultiTaskNetwork, self).__init__()
        self.shared = gluon.nn.HybridSequential()
        with self.shared.name_scope():
            self.shared.add(
                gluon.nn.Dense(128, activation='relu'),
                gluon.nn.Dense(64, activation='relu'),
                gluon.nn.Dense(10, activation='relu')
            )
        self.output1 = gluon.nn.Dense(10)  # Digit
        self.output2 = gluon.nn.Dense(1)  # Odd Even

    def hybrid_forward(self, F, x, *args, **kwargs):
        """

        :param F:
        :param x:
        :param args:
        :param kwargs:
        :return: output_digit with shape (batch_size, 10) - Dense(10) and output_odd_even with shape (batch_size,
        1) - Dense(1)
        """
        y = self.shared(x)
        output1 = self.output1(y)
        output2 = self.output2(y)
        return output1, output2


loss_digits = gluon.loss.SoftmaxCELoss()
loss_odd_even = gluon.loss.SigmoidBCELoss()

mx.random.seed(42)
random.seed(42)

net = MultiTaskNetwork()
net.initialize(mx.init.Xavier(), ctx=ctx)

net.hybridize()

trainer = gluon.Trainer(net.collect_params(), optimizer='adam',
                        optimizer_params={'learning_rate': lr})


def evaluate_accuracy(net, data_iterator):
    acc_digits = mx.metric.Accuracy(name='digits')
    acc_odd_even = mx.metric.Accuracy(name='odd_even')

    # data - (batch_size, 1, num_features, num_features)
    # label_digit - (batch_size, )
    # label_odd_even - (batch_size, ) with 0, 1 (0 is even, odd is 1)
    for i, (data, label_digit, label_odd_even) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label_digit = label_digit.as_in_context(ctx)
        label_odd_even = label_odd_even.as_in_context(ctx).reshape(-1, 1)
        output_digit, output_odd_even = net(data)

        acc_digits.update(label_digit, output_digit.softmax())
        acc_odd_even.update(label_odd_even, output_odd_even.sigmoid() > 0.5)
    return acc_digits.get(), acc_odd_even.get()


if __name__ == '__main__':
    alpha = 0.5
    for e in range(epochs):
        acc_digits = mx.metric.Accuracy(name='digits')
        acc_odd_even = mx.metric.Accuracy(name='odd_even')
        l_digits_ = 0.
        l_odd_even_ = 0.
        i = 0

        for i, (data, label_digit, label_odd_even) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label_digit = label_digit.as_in_context(ctx)
            label_odd_even = label_odd_even.as_in_context(ctx).reshape(-1, 1)

            with autograd.record():
                # output_digit - (batch_size, 10), output_odd_even - (batch_size, )
                output_digit, output_odd_even = net(data)
                l_digits = loss_digits(output_digit, label_digit)
                l_odd_even = loss_odd_even(output_odd_even, label_odd_even)

                l_combined = (1 - alpha) * l_digits + alpha * l_odd_even

            l_combined.backward()
            trainer.step(data.shape[0])

            l_digits_ += l_digits.mean()
            l_odd_even_ += l_odd_even.mean()
            acc_digits.update(label_digit, output_digit.softmax())
            acc_odd_even.update(label_odd_even, output_odd_even.sigmoid() > 0.5)
        print("Epoch [{}], Acc Digits\t{:.4f} Loss Digits\t{:.4f}"
              .format(e, acc_digits.get()[1], l_digits_.asscalar() / (i + 1)))
        print("Epoch [{}], Acc Odd/Even {:.4f} Loss Odd/Even {:.4f}".format(
            e, acc_odd_even.get()[1], l_odd_even_.asscalar() / (i + 1)))
        print("Epoch [{}], Testing Accuracies {}".format(e, evaluate_accuracy(net, test_data)))



