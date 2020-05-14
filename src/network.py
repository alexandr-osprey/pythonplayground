import numpy as np

class Network:
    def __init__(self, activation, output_activation, error, layer_dims, problem_data, epochs, learning_rate):
        self.W = []
        self.B = []
        self.error = error
        self.layer_dims = layer_dims
        self.L = len(layer_dims)
        self.problem_data = problem_data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.test_cost = 100
        self.activations = [activation] * (self.L - 2) + [output_activation]

    def _initialize_parameters(self):
        for l in range(1, self.L):
            n = self.layer_dims[l]
            p = self.layer_dims[l - 1]
            w = np.random.randn(n, p) - 0.2
            b = np.zeros((n, 1))
            self.W.append(w)
            self.B.append(b)

    def _forward_prop(self, X):
        A_prev = X
        z_cache = []
        a_cache = []
        a_cache.append(X)
        for l in range(self.L - 1):
            z = self.W[l].dot(A_prev) + self.B[l]
            A = self.activations[l].func(z)
            z_cache.append(z)
            a_cache.append(A)
            A_prev = A
        
        return A, z_cache, a_cache

    def _back_prop(self, AL, Y, z_cache, a_cache):
        AL.reshape(Y.shape)
        dA_prev = self.error.loss_der(AL, Y)
        dB = []
        dW = []
        m = len(Y)
        for l in reversed(range(self.L - 1)):
            dZ = dA_prev * self.activations[l].der(z_cache[l])
            A_prev = a_cache[l]
            dW.append(1./m * np.dot(dZ, A_prev.T))
            dB.append(1./m * np.sum(dZ, axis=1, keepdims=True))
            dA_prev = np.dot(self.W[l].T, dZ)
            
        dW.reverse()
        dB.reverse()
        return dW, dB

    def _update_parameters(self, dW, dB, learning_rate):
        for l in range(self.L - 1):
            self.W[l] = self.W[l] - dW[l] * learning_rate
            self.B[l] = self.B[l] - dB[l] * learning_rate

    def build_and_train(self):
        train_data, test_data = self.problem_data
        self._initialize_parameters()
        W_history, B_history, test_cost_history = [], [], []
        for e in range(self.epochs):
            AL, z_cache, a_cache = self._forward_prop(train_data[0])
            dW, dB = self._back_prop(AL, train_data[1], z_cache, a_cache)
            _, cost_test = self.get_cost(test_data[0], test_data[1])
            W_history.append(self.W)
            B_history.append(self.B)
            test_cost_history.append(cost_test)
            self._update_parameters(dW, dB, self.learning_rate)

        min_i = np.argmin(test_cost_history)
        self.W, self.B, self.test_cost = W_history[min_i], B_history[min_i], test_cost_history[min_i]

    def get_cost(self, X, Y):
        AL, _, _ = self._forward_prop(X)
        cost = self.error.cost(AL, Y)
        return AL, cost