import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

num_features = 784  # data features (img shape: 28*28).

# Training parameters.
learning_rate = 0.01
training_steps = 20000
batch_size = 256
display_step = 1000

# Network Parameters
num_hidden_1 = 128  # 1st layer num features.
num_hidden_2 = 64  # 2nd layer num features (the latent dim).

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255.0, x_test / 255.0

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.repeat().batch(batch_size).prefetch(1)

random_normal = tf.initializers.RandomNormal()

weights = {
    'encoder_h1': tf.Variable(random_normal([num_features, num_hidden_1])),
    'encoder_h2': tf.Variable(random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(random_normal([num_hidden_1, num_features]))
}
biases = {
    'encoder_b1': tf.Variable(random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(random_normal([num_features]))
}


def encoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encoder_h1"]),
                                  biases["encoder_b1"]))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights["encoder_h2"]),
                                  biases["encoder_b2"]))
    return layer2


def decoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decoder_h1"]),
                                  biases["decoder_b1"]))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights["decoder_h2"]),
                                  biases["decoder_b2"]))
    return layer2


def mean_square(reconstructed, original):
    return tf.reduce_mean(tf.pow(original - reconstructed, 2))


optimizer = tf.optimizers.Adam(learning_rate)


def run_optimization(x):
    with tf.GradientTape() as g:
        reconstructed_image = decoder(encoder(x))
        loss = mean_square(reconstructed_image, x)

    trainable_variables = [*weights.values()] + [*biases.values()]

    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss


for step, (batch_x, _) in enumerate(train_data.take(training_steps + 1)):
    loss = run_optimization(batch_x)
    if step % display_step == 0:
        print("step: %i, loss: %.3f" % (step, loss))

n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
for i, (batch_x, _) in enumerate(train_data.take(n)):
    reconstructed_images = decoder(encoder(batch_x))
    for j in range(n):
        img = batch_x[j].numpy().reshape([28, 28])
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img
    for j in range(n):
        reconstructed_img = reconstructed_images[j].numpy().reshape([28, 28])
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = reconstructed_img

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin='upper', cmap="Greys")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="Greys")
plt.show()
