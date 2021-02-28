import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from tensorflow.keras.datasets import mnist

num_classes = 10  # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

# Network parameters.
conv1_filters = 32  # number of filters for 1st conv layer.
conv2_filters = 64  # number of filters for 2nd conv layer.
fc1_units = 1024  # number of neurons for 1st fully-connected layer.


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


class ConvNet(Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)

        self.max_pool1 = layers.MaxPool2D(2, strides=2)

        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)

        self.max_pool2 = layers.MaxPool2D(2, strides=2)

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(1024)

        self.dropout = layers.Dropout(.5)

        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False, *args):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x


def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


conv_net = ConvNet()
optimizer = tf.optimizers.Adam(learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = conv_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)

    trainable_variables = conv_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Step: %i, loss: %.4f, acc: %.4f" % (step, loss, acc))


