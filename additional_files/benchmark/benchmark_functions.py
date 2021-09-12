import numpy as np


'''
x^2 
Key characteristics:
* Single variable real valued function
* Range is non negative
* Convex
'''
def x_squared(x: np.array) -> np.float32:
    return x * x

'''
1/(1+e^(-x))
Key characteristics:
* Single variable real valued function
* Range continuous interval of [0,1]
* Commonly used neural network activation function
'''
def sigmoid(x: np.array) -> np.float32:
    return 1/(1 + np.exp(-x))

'''
Key characteristics
* Multi-variable real valued function
* Range is non negative
* Convex
'''
def sphere(x: np.array) -> np.float32:
    return np.sum(x * x)

'''
Key characteristics
* 2 dimensional function used for optimization benchmarking
* Highly non convex
* Single global minimum at (0,0)
src: https://en.wikipedia.org/wiki/Ackley_function
'''
def ackley(x: np.array) -> float:
    assert(len(x) == 2)
    x1 = x[0]
    x2 = x[1]
    p1 = -0.2 * np.sqrt(0.5 * (x1 * x1 + x2 * x2))
    p2 = 0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))
    return np.exp(1) + 20 - 20 * np.exp(p1) - np.exp(p2)

'''
Key characteristics
* n dimensional function used for optimization benchmarking
* Highly non convex
* Global minimum at (0, ..., 0)
* Common search domain, -5.12 <= x_i <= 5.12
* https://en.wikipedia.org/wiki/Test_functions_for_optimization
'''
def rastrigin(x: np.array) -> float:
    A = 10
    ret = A * len(x)
    for t in x:
        ret += t * t - A * np.cos(2 * np.pi * t)
    return ret

def booth(x: np.array) -> float:
    assert(len(x) == 2)
    x1 = x[0]
    x2 = x[1]