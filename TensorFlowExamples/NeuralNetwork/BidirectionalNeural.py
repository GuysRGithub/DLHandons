import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from tensorflow.keras.datasets import mnist

num_classes = 10  # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).

# Training Parameters
learning_rate = 0.001
training_steps = 1000
batch_size = 32
display_step = 100

# Network Parameters
# MNIST image shape is 28*28px, we will then handle 28 sequences of 28 timesteps for every sample.
num_input = 28  # number of sequences.
timesteps = 28  # timesteps.
num_units = 32  # number of neurons for the LSTM layer.


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)

x_train, x_test = x_train.reshape([-1, 28, 28]), x_test.reshape([-1, 28, 28])
x_train, x_test = x_train / 255.0, x_test / 255.0


train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


class BiRNN(Model):
    def __init__(self):
        super(BiRNN, self).__init__()
        lstm_fw = layers.LSTM(units=num_units)
        lstm_bw = layers.LSTM(units=num_units, go_backwards=True)

        self.bi_lstm = layers.Bidirectional(lstm_fw, backward_layer=lstm_bw)
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False, *args):
        x = self.bi_lstm(x)
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


optimizer = tf.optimizers.Adam(learning_rate)
biRrn_net = BiRNN()


def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = biRrn_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)

    trainable_variables = biRrn_net.trainable_variables

    gradients = g.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))


for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = biRrn_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Step: %i, loss: %.4f, acc: %.4f" % (step, loss, acc))

