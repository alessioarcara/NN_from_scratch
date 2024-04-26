import numpy as np
from activation_functions import activations, activation_derivs
from weight_initializers import weight_initializers
from scipy.linalg import toeplitz


class Layer:
    def __init__(self):
        self.x = None
        self.a = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, d):
        raise NotImplementedError


class Dense(Layer):
    def __init__(
            self,
            input_size,
            output_size,
            initialization="he",
            activation="relu"
    ):
        super().__init__()
        self.w = weight_initializers[initialization](input_size, output_size)
        self.b = np.zeros((output_size, 1))
        self.activation = activations[activation]
        self.activation_deriv = activation_derivs[activation]

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.w, self.x) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, d, lr):
        m = d.shape[1]
        d = d * self.activation_deriv(self.z)
        dw = np.dot(d, self.x.T) / m
        db = np.mean(d, axis=1, keepdims=True)
        self.w -= lr * dw
        self.b -= lr * db
        return np.dot(self.w.T, d)
    

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if len(x.shape) > 2:
            return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        else:
            return x

    def backward(self, d, lr):
        return d

    
class Conv(Layer):
    def __init__(
        self, 
        n_kernels,
        input_shape, 
        kernel_shape, 
        activation="relu"
    ):
        super().__init__()
        self.activation = activations[activation]
        self.activation_deriv = activation_derivs[activation]
        
        self.kernels = []
        for _ in range(n_kernels):
            kernel = np.random.rand(*kernel_shape)
            self.kernels.append(kernel)
        
        self.toeplitz_matrices = []
        for kernel in self.kernels:
            toeplitz_matrix = self.conv2d_to_toeplitz(input_shape, kernel)
            self.toeplitz_matrices.append(toeplitz_matrix)
        
        self.kernels = np.array(self.kernels)
        self.toeplitz_matrices = np.array(self.toeplitz_matrices)
            
    def conv2d_to_toeplitz(self, input_shape, kernel):
        ih, iw = input_shape
        kh, kw = kernel.shape

        K = np.concatenate([
            toeplitz(
                c=(kernel[r,0], *np.zeros(iw-kw)),
                r=(*kernel[r], *np.zeros(iw-kw))
            )
            for r in range(kh)
        ], axis=1)

        h = K.shape[0]
        M = np.concatenate(
        [
            np.concatenate([
                np.zeros((h, iw*c)),
                K,
                np.zeros((h, iw*(ih-kh-c)))
            ], axis=1)
            for c in range(ih-kh+1)
        ],
        axis=0)

        return M
        
    
    def forward(self, x):
        self.x = x
        self.z = np.dot(self.toeplitz_matrices, self.x)
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, d, lr):
        pass
