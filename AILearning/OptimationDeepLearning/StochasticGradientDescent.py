from d2l import AllDeepLearning as d2l
import math
import numpy as np
from AI.AILearning.OptimationDeepLearning import GradientDescent as g
"""
    Function
        f(x)=1n∑i=1nfi(x).
    
        Gradient
    ∇f(x)=1n∑i=1n∇fi(x).
    
    Stochastic gradient descent (SGD) reduces computational cost at each iteration.
    At each iteration of stochastic gradient descent, we uniformly sample
    an index  i∈{1,…,n}  for data instances at random, and compute the 
    gradient  ∇fi(x)  to update  x :
        x←x−η∇fi(x).
        
    We should mention that the stochastic gradient ∇fi(x) is the 
    unbiased estimate of gradient ∇f(x).
        Ei∇fi(x)=1n∑i=1n∇fi(x)=∇f(x).
    
"""


def f(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2  # Objective


def gradf(x1, x2):
    return 2 * x1, 4 * x2  # Gradient


def sgd(x1, x2, s1, s2):  # Simulate noisy gradient
    global lr  # Learning rate scheduler
    (g1, g2) = gradf(x1, x2)  # Compute gradient
    (g1, g2) = (g1 + np.random.normal(0.1), g2 + np.random.normal(0.1))
    eta_t = eta * lr()  # Learning rate at time t
    return x1 - eta_t * g1, x2 - eta_t * g2, 0, 0  # Update variables


eta = 0.1


# lr = (lambda: 1)  # Constant learning rate
def exponential():
    global ctr
    ctr += 1
    return math.exp(-0.1 * ctr)


def polynomial():
    global ctr
    ctr += 1
    return (1 + 0.1 * ctr) ** (-0.5)


ctr = 1
lr = polynomial
print(lr)
d2l.show_trace_2d(f, g.train_2d(sgd, steps=50))
d2l.plt.show()

"""
    ##########################                SUMMARY                         #########################
    For convex problems we can prove that for a wide choice of learning 
    rates Stochastic Gradient Descent will converge to the optimal solution.

    For deep learning this is generally not the case. However, 
    the analysis of convex problems gives us useful insight into how to 
    approach optimization, namely to reduce the learning rate progressively, 
    albeit not too quickly.

    Problems occur when the learning rate is too small or too large. 
    In practice a suitable learning rate is often found only after multiple experiments.

    When there are more examples in the training dataset, it costs more to 
    compute each iteration for gradient descent, so SGD is preferred in these cases.

    Optimality guarantees for SGD are in general not available in 
    nonconvex cases since the number of local minima that require checking 
    might well be exponential. 
"""
