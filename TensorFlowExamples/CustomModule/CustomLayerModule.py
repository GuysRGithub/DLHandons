import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras.datasets import mnist

num_classes = 10  # 0 to 9 digits
num_features = 784  # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = 500
batch_size = 256
display_step = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])

x_train, x_test = x_train / 255.0, x_test / 255.0

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


class CustomLayer(layers.Layer):
    def __init__(self, num_units, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.num_units = num_units

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.num_units))
        self.weight = self.add_weight(name="W",
                                      shape=shape,
                                      initializer=tf.initializers.RandomNormal,
                                      trainable=True)
        self.bias = self.add_weight(name='b',
                                    shape=[self.num_units])
        super(CustomLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.matmul(inputs, self.weight)
        x = x + self.bias
        return tf.nn.relu(x)

    def get_config(self):
        base_config = super(CustomLayer, self).get_config()
        base_config['num_units'] = self.num_units
        return base_config


class AnotherCustomLayer(layers.Layer):
    def __init__(self, num_units, **kwargs):
        super(AnotherCustomLayer, self).__init__(**kwargs)
        self.num_units = num_units

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.num_units))
        self.inner_layer1 = layers.Dense(1)
        self.inner_layer2 = layers.Dense(self.num_units)

        super(AnotherCustomLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.inner_layer1(inputs)
        x = tf.nn.relu(x)
        x = self.inner_layer2(x)
        return x + inputs

    def get_config(self):
        base_config = super(AnotherCustomLayer, self).get_config()
        base_config['num_units'] = self.num_units
        return base_config


class CustomNet(models.Model):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.layer1 = CustomLayer(64)
        self.layer2 = AnotherCustomLayer(64)
        self.out = layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, x, is_training=False, *args):
        x = self.layer1(x)
        x = tf.nn.relu(x)
        x = self.layer2(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x


def cross_entropy(y_pred, y_true):
    y_true = tf.cast(y_true, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


customNet = CustomNet()
optimizer = tf.optimizers.Adam(learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = customNet(x)
        loss = cross_entropy(pred, y)

    gradients = g.gradient(loss, customNet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, customNet.trainable_variables))


for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = customNet(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Step: %i, loss: %.3f, acc: %.3f" % (step, loss, acc))


customNet = CustomNet()
