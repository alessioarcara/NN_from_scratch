import numpy as np
from scipy.special import expit


def sigmoid(z):
    return expit(z)


def dsigmoid(z):
    s = sigmoid(z)
    return s * (1. - s)


def relu(z):
    return np.maximum(0, z)


def drelu(z):
    return np.where(z > 0, 1, 0)


activations = { "relu": relu, "sigmoid": sigmoid }
activation_derivs = { "relu": drelu, "sigmoid": dsigmoid }