import numpy as np


def xavier(input_size, output_size):
    return np.random.randn(output_size, input_size) * np.sqrt(2 / (output_size + input_size))


def he(input_size, output_size):
    return np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)


weight_initializers = { "xavier": xavier, "he": he }
