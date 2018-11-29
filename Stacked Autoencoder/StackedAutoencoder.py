import numpy as np
import argparse
from load_fashion_mnist import load_fashion_mnist, separate_train_val_data
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from SparseAutoencoder import SparseAutoencoder, sigmoid
from time import time


class StackedAutoencoder:
    def __init__(self, input_size=784, hidden_layer_sizes=[500, 250, 100], output_size=10, max_iterations=1000, method='L-BFGS-B'):
        self.max_iterations = max_iterations  # number of optimization iterations
        self.network_dims = [input_size] + hidden_layer_sizes + [output_size]
        self.method = method

        self.rho = 0.1  # desired average activation of hidden units
        self.lamda = 0.003  # weight decay parameter
        self.beta = 3  # weight of sparsity penalty term

        self.list_auto_encoders = []
        self.packed_auto_encoder_stack_with_softmax = []

    def softmaxCost(self, params, X, labels):

        params = params.reshape(self.network_dims[-1], self.network_dims[-2])
        hypothesis = np.exp(np.dot(params, X))
        probs = hypothesis / np.sum(hypothesis, axis=0)

        ground_truth = getGroundTruth(labels)
        weight_decay = 0.5 * self.lamda * np.sum(np.multiply(params, params))
        cost = np.multiply(ground_truth, np.log(probs))
        cost = -(np.sum(cost) / X.shape[1]) + weight_decay

        gradient = -np.dot(ground_truth - probs, X.T)
        gradient = gradient / X.shape[1] + self.lamda * params
        gradient = np.array(gradient).flatten()

        return [cost, gradient]

    def predict(self, X, classifier, weights_biases=None):
        classifier = classifier.reshape(self.network_dims[-1], self.network_dims[-2])
        activation = X
        if weights_biases is None:
            for autoencoder in self.list_auto_encoders:
                activation = sigmoid(np.dot(autoencoder.W1, activation) + autoencoder.b1)
        else:
            start = 0
            for autoencoder in self.list_auto_encoders:
                W_limits = (start, start + autoencoder.hidden_size * autoencoder.input_size)
                b_limits = (W_limits[1], W_limits[1] + autoencoder.hidden_size)
                W = weights_biases[W_limits[0]: W_limits[1]].reshape(autoencoder.hidden_size, autoencoder.input_size)
                b = weights_biases[b_limits[0]: b_limits[1]].reshape(autoencoder.hidden_size, 1)
                start = b_limits[1]
                activation = sigmoid(np.dot(W, activation) + b)
        hypothesis = np.exp(np.dot(classifier, activation))
        probs = hypothesis / np.sum(hypothesis, axis=0)
        preds = np.zeros((X.shape[1], 1))
        preds[:, 0] = np.argmax(probs, axis=0)
        return preds

    def packSoftmaxWeightsAndBiases(self, softmax_pack=None, weights_biases=None):
        packed = []
        if not softmax_pack is None:
            packed = np.concatenate((packed, np.array(softmax_pack).flatten()))
        if weights_biases is None:
            for auto_encoder in self.list_auto_encoders:
                packed = np.concatenate((packed, np.array(auto_encoder.W1).flatten(), np.array(auto_encoder.b1).flatten()))
        else:
            for (w, b) in weights_biases:
                packed = np.concatenate((packed, np.array(w).flatten(), np.array(b).flatten()))
        return packed

    def unpackSoftmaxWeightsAndBiases(self, pack):
        softmax_size = self.network_dims[-1] * self.network_dims[-2]
        classifier = pack[0: softmax_size].reshape(self.network_dims[-1], self.network_dims[-2])
        ae_start, unpacked_wb, packed_stack = 0, [], pack[softmax_size:]
        for ae in self.list_auto_encoders:
            index_W = ae_start + ae.input_size * ae.hidden_size
            index_b = index_W +  ae.hidden_size
            W = packed_stack[ae_start:index_W].reshape(ae.hidden_size, ae.input_size)
            b = packed_stack[index_W:index_b].reshape(ae.hidden_size, 1)
            ae_start = index_b
            unpacked_wb.append((W, b))
        return classifier, unpacked_wb

    def findAccuracy(self, X, Y, min_func_name, min_x0, **kwargs):
        solution = minimize(min_func_name, min_x0, args=kwargs.values()[0], method=self.method, jac=True,
                            options={'maxiter': self.max_iterations, 'disp': True})
        solution = solution.x
        if 'StackedAutoencoder.softmaxCost' in str(min_func_name):
            self.packed_auto_encoder_stack_with_softmax = self.packSoftmaxWeightsAndBiases(solution)
            prediction = self.predict(X, solution)
        else:
            classifier_length = self.network_dims[-1] * self.network_dims[-2]
            prediction = self.predict(X, solution[0:classifier_length], solution[classifier_length:])
        test_accuracy = Y[:, 0] == prediction[:, 0]
        return np.mean(test_accuracy)

    def sample_finetuning_data(self, trX, trY, samples_per_class=1):
        total_data_size, no_of_classes = trX.shape[-1], self.network_dims[-1]
        no_val_per_class = (total_data_size - (no_of_classes * samples_per_class)) / no_of_classes
        ftX, ftY, _, _ = separate_train_val_data(trX.shape[1], samples_per_class, no_val_per_class, no_of_classes, trX, trY)
        return ftX, ftY

    def runStackedAutoencoder(self, trX, trY, tsX, tsY):
        i, activation = 1, trX
        while i < len(self.network_dims) - 1:
            sparse_encoder = SparseAutoencoder(self.network_dims[i-1], self.network_dims[i], self.rho, self.lamda, self.beta)
            packed_inputs = sparse_encoder.packWeightsBiases()
            opt_packed_params = minimize(fun=sparse_encoder.cost, x0=packed_inputs, args=(activation, ), jac=True,
                                            method=self.method, options={'maxiter': self.max_iterations, 'disp': True})
            sparse_encoder.unpackWeightsBiases(opt_packed_params.x)
            self.list_auto_encoders.append(sparse_encoder)
            activation = 1. / (1 + np.exp(-(np.dot(sparse_encoder.W1, activation) + sparse_encoder.b1)))
            i += 1

        rand = np.random.RandomState(np.random.seed(int(time())))
        softmax_x0 = 0.005 * np.asarray(rand.normal(size=(self.network_dims[-1]*self.network_dims[-2], 1)))
        trAcc = self.findAccuracy(tsX, tsY, self.softmaxCost, softmax_x0, accuracy_type='training & softmax',
                                         kwargs=(activation, trY))
        print "Accuracy after training & softmax = {0}".format(trAcc)


        ftX_5, ftY_5 = self.sample_finetuning_data(trX, trY, samples_per_class=5)
        ftX_1, ftY_1 = self.sample_finetuning_data(ftX_5, ftY_5, samples_per_class=1)
        ftAcc1 = self.findAccuracy(tsX, tsY, self.stackedAutoencoderCost, self.packed_auto_encoder_stack_with_softmax,
                                      kwargs=(ftX_1, ftY_1))
        ftAcc5 = self.findAccuracy(tsX, tsY, self.stackedAutoencoderCost, self.packed_auto_encoder_stack_with_softmax,
                                      kwargs=(ftX_5, ftY_5))
        print "Accuracy after fine-tuning with 1-labeled samples per class = {0}".format(ftAcc1)
        print "Accuracy after fine-tuning with 5-labeled samples per class = {0}".format(ftAcc5)

    def stackedAutoencoderCost(self, packed_stack_with_softmax, X, Y):

        classifier, unpacked_wb = self.unpackSoftmaxWeightsAndBiases(packed_stack_with_softmax)
        """ Calculate activations for every layer """
        layers_count = len(unpacked_wb)
        activation = {0: X}
        for i in range(layers_count):
            activation[i + 1] = sigmoid(np.dot(unpacked_wb[i][0], activation[i]) + unpacked_wb[i][1])

        ground_truth = getGroundTruth(Y)
        hypothesis = np.exp(np.dot(classifier, activation[layers_count]))
        probs = hypothesis / np.sum(hypothesis, axis=0)

        cost = np.multiply(ground_truth, np.log(probs))
        cost = -(np.sum(cost) / X.shape[1])
        cost += 0.5 * self.lamda * np.sum(np.multiply(classifier, classifier))  # weight decay

        classifier_gradient = -np.dot(ground_truth - probs, np.transpose(activation[layers_count]))
        classifier_gradient = classifier_gradient / X.shape[1] + self.lamda * classifier

        delta = {layers_count: -np.multiply(np.dot(classifier.T, ground_truth - probs),
                                            np.multiply(activation[layers_count], 1 - activation[layers_count]))}
        for i in range(layers_count - 1):
            index = layers_count - i - 1
            delta[index] = np.multiply(np.dot(unpacked_wb[index][0].T, delta[index + 1]),
                                       np.multiply(activation[index], 1 - activation[index]))

        wb_gradient = [None] * layers_count
        for layer_index in range(layers_count):
            i = layers_count - layer_index - 1
            w = np.dot(delta[i + 1], activation[i].T) / X.shape[1]
            b = np.sum(delta[i + 1], axis=1) / X.shape[1]
            wb_gradient[i] = (w, b)
        gradient = self.packSoftmaxWeightsAndBiases(classifier_gradient, wb_gradient)
        return [cost, gradient]


def getGroundTruth(Y):
    Y = np.array(Y).flatten()
    ones = np.ones(len(Y))
    indexes = np.arange(len(Y) + 1)
    ground_truth = csr_matrix((ones, Y, indexes)).todense().T
    return ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--hidden', type=str, default="[500, 200, 100]")
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-m', '--method', type=str, default='L-BFGS-B')
    parser.add_argument('-i', '--input', type=int, default=784)
    parser.add_argument('-o', '--output', type=int, default=10)
    args = parser.parse_args()
    hidden_layers = [int(h) for h in args.hidden.strip('[').strip(']').split(',' if ',' in args.hidden else ' ')]
    trX, trY, tsX, tsY = load_fashion_mnist()
    sae = StackedAutoencoder(input_size=args.input, hidden_layer_sizes=hidden_layers, output_size=args.output,
                             max_iterations=args.epochs, method=args.method)
    sae.runStackedAutoencoder(trX, trY, tsX, tsY)


if __name__ == "__main__":
    main()