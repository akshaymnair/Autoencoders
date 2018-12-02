import numpy as np
from HelperFunctions import sigmoid


class SparseAutoencoder:
    def __init__(self, input_size, hidden_size, rho, lamda, beta):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rho = rho
        self.lamda = lamda
        self.beta = beta
        rand = np.random.RandomState(1234)
        dist_range = np.sqrt(5) / np.sqrt(input_size + hidden_size)
        self.W1 = np.array(rand.uniform(-dist_range, dist_range, (hidden_size, input_size)))
        self.W2 = np.array(rand.uniform(-dist_range, dist_range, (input_size, hidden_size)))
        self.b1 = np.zeros((hidden_size, 1))
        self.b2 = np.zeros((input_size, 1))
        self.features = None
        self.costs = []

    def packWeightsBiases(self, pack_self=True, W1=None, W2=None, b1=None, b2=None):
        if pack_self:
            W1, W2, b1, b2 = self.W1, self.W2, self.b1, self.b2
        return np.concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))

    def unpackWeightsBiases(self, packed_weights_biases, self_assign=True):
        W1_limits = (0, self.hidden_size * self.input_size)
        W2_limits = (W1_limits[1], W1_limits[1] * 2)
        b1_limits = (W2_limits[1], W2_limits[1] + self.hidden_size)
        b2_limits = (b1_limits[1], b1_limits[1] + self.input_size)
        W1 = packed_weights_biases[W1_limits[0]: W1_limits[1]].reshape(self.hidden_size, self.input_size)
        W2 = packed_weights_biases[W2_limits[0]: W2_limits[1]].reshape(self.input_size, self.hidden_size)
        b1 = packed_weights_biases[b1_limits[0]: b1_limits[1]].reshape(self.hidden_size, 1)
        b2 = packed_weights_biases[b2_limits[0]: b2_limits[1]].reshape(self.input_size, 1)
        if self_assign:
            self.W1, self.W2, self.b1, self.b2 = W1, W2, b1, b2
        return W1, W2, b1, b2

    def fit(self, packed_weights_biases, X):
        W1, W2, b1, b2 = self.unpackWeightsBiases(packed_weights_biases, self_assign=False)

        hidden_activation = sigmoid(np.dot(W1, X) + b1)
        output_activation = sigmoid(np.dot(W2, hidden_activation) + b2)
        rho_cap = np.sum(hidden_activation, axis=1) / X.shape[1]
        delta = output_activation - X
        sse = 0.5 * np.sum(np.multiply(delta, delta)) / X.shape[1]
        decay = 0.5 * self.lamda * (np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)))
        KL_divergence = self.beta * np.sum(self.rho * np.log(self.rho / rho_cap) +
                                           (1 - self.rho) * np.log((1 - self.rho) / (1 - rho_cap)))
        cost = sse + decay + KL_divergence
        self.costs.append(cost)

        gradient_KL_divergence = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        delta_output = np.multiply(delta, np.multiply(output_activation, 1 - output_activation))
        delta_hidden = np.multiply(np.dot(np.transpose(W2), delta_output) + np.transpose(np.matrix(gradient_KL_divergence)),
                              np.multiply(hidden_activation, 1 - hidden_activation))

        gradient_W1 = np.dot(delta_hidden, np.transpose(X))
        gradient_W1 = np.array(gradient_W1 / X.shape[1] + self.lamda * W1)
        gradient_W2 = np.dot(delta_output, np.transpose(hidden_activation))
        gradient_W2 = np.array(gradient_W2 / X.shape[1] + self.lamda * W2)
        gradient_b1 = np.array(np.sum(delta_hidden, axis=1) / X.shape[1])
        gradient_b2 = np.array(np.sum(delta_output, axis=1) / X.shape[1])

        packed_gradients = self.packWeightsBiases(pack_self=False, W1=gradient_W1, W2=gradient_W2,
                                                  b1=gradient_b1, b2=gradient_b2)
        return [cost, packed_gradients]
