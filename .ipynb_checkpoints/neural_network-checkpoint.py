import numpy as np
from tqdm import tqdm


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, d, lr):
        for layer in reversed(self.layers):
            d = layer.backward(d, lr)

    def fit(self, X, y, n_epochs, lr=0.01, mini_batch_size=32, val_data=None, verbose=False):
        n = len(X)
        indices = np.arange(n)

        for epoch in range(n_epochs):
            if verbose:
                self.print_params()
            np.random.shuffle(indices)
            mini_batch_indices = [indices[k:k+mini_batch_size]
                                  for k in range(0, n, mini_batch_size)]

            for mini_batch_idx in tqdm(mini_batch_indices, colour='GREEN'):
                X_batch = np.atleast_2d(X[mini_batch_idx]).T
                y_batch = np.atleast_2d(y[mini_batch_idx]).T

                a = self.forward(X_batch)
                d = (a - y_batch)
                self.backward(d, lr)

            print(f"Epoch {epoch + 1}/{n_epochs}")
            if val_data:
                X_val, y_val = val_data
                y_val_pred = np.argmax(self.predict(X_val), axis=0)
                val_acc = np.mean(y_val_pred == y_val)
                print(f"Val acc: {val_acc:.4f}")
                
    def predict(self, X):
        X = np.atleast_2d(X).T
        return self.forward(X)

    def print_params(self):
        print("----------------------------")
        print("Network Parameters:")
        print("----------------------------")

        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1} - Weights:")
            print(layer.w)
            print()
            print("Biases:")
            print(layer.b.reshape(-1, 1))
            print()

