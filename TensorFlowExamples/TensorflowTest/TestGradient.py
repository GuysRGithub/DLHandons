import tensorflow as tf


with tf.GradientTape() as g:
    x = tf.constant([1.0, 3.0, 5.0])
    y = tf.constant([2.0, 4.0, 6.0])
    z = x ** 2 + y ** 2
    g.watch(z)
    t = z ** 2
jacobian = g.jacobian(t, z)
print(z)
print(jacobian)