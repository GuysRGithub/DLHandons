from IPython import display
from d2l import AllDeepLearning as d2l
from mxnet import nd, gluon
import numpy as np

path = "E:/Python_Data"


def angle(v, w):
    return np.arccos(np.dot(v, w)) / (np.linalg.norm(v) * np.linalg.norm(w))


train = gluon.data.vision.FashionMNIST(path, train=True)
test = gluon.data.vision.FashionMNIST(path, train=False)

X_train_0 = nd.stack(*[x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = nd.stack(*[x[0] for x in train if x[1] == 1]).astype(float)

X_test = nd.stack(*[x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

y_test = nd.stack(
    *[nd.array([x[1]]) for x in test if x[1] == 0 or x[1] == 1]).astype(float)

ave_0 = nd.mean(X_train_0, axis=0)
ave_1 = nd.mean(X_train_1, axis=0)
# print(ave_0)
# print("WTF")
d2l.set_figsize()
d2l.plt.imshow(ave_1.reshape(28, 28).asnumpy(), cmap="Greys")
d2l.plt.show()

w = (ave_1 - ave_0).reshape(-1, 1)
predictions = nd.dot(X_test.reshape(2000, -1), w.flatten()) > -1500000

print(nd.mean(predictions.astype(y_test.dtype) == y_test))


"""
    Vectors can be interpreted geometrically as either points or directions in space.

    Dot products define the notion of angle to arbitrarily high-dimensional spaces.

    Hyperplanes are high-dimensional generalizations of lines and planes.
    They can be used to define decision planes that are often used as the last 
    step in a classification task.

    Matrix multiplication can be geometrically interpreted as 
    uniform distortions of the underlying coordinates. They represent a
    very restricted, but mathematically clean, way to transform vectors.

    Linear dependence is a way to tell when a collection of vectors are in a 
    lower dimensional space than we would expect (say you have  3  vectors living
    in a  2 -dimensional space). The rank of a matrix is the size of the largest subset 
    of its columns that are linearly independent.

    When a matrix’s inverse is defined, matrix inversion allows us to find 
    another matrix that undoes the action of the first. Matrix inversion is 
    useful in theory, but requires care in practice owing to numerical instability.

    Determinants allow us to measure how much a matrix expands or contracts a space. 
    A nonzero determinant implies an invertible (non-singular) matrix and a zero-valued 
    determinant means that the matrix is non-invertible (singular).

    Tensor contractions and Einstein summation provide for a neat and 
    clean notation for expressing many of the computations that are seen in machine learning.


"""