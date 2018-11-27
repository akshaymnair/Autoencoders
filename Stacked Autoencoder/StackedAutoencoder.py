import numpy as np
import scipy.io
from load_fashion_mnist import load_fashion_mnist, separate_train_val_data
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from SparseAutoencoder import SparseAutoencoder, sigmoid
from time import time
# from stackedAutoencoder_1 import SparseAutoencoder


class StackedAutoencoder:
    def __init__(self, input_size=784, hidden_layer_sizes=[500, 250, 100], output_size=10, max_iterations=1000):
        self.max_iterations = max_iterations  # number of optimization iterations
        self.network_dims = [input_size] + hidden_layer_sizes + [output_size]

        self.rho = 0.1  # desired average activation of hidden units
        self.lamda = 0.003  # weight decay parameter
        self.beta = 3  # weight of sparsity penalty term

        self.list_auto_encoders = []

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

    def predict(self, result, X):
        activation = X
        for autoencoder in self.list_auto_encoders:
            activation = sigmoid(np.dot(autoencoder.W1, activation) + autoencoder.b1)
        hypothesis = np.exp(np.dot(result, activation))
        probs = hypothesis / np.sum(hypothesis, axis=0)
        preds = np.zeros((X.shape[1], 1))
        preds[:, 0] = np.argmax(probs, axis=0)
        return preds

    def packSoftmaxWeightsAndBiases(self, softmax_pack, weights_biases=None):
        packed = np.concatenate(([], softmax_pack.flatten()))
        if not weights_biases:
            for auto_encoder in self.list_auto_encoders:
                packed = np.concatenate((packed, auto_encoder.W1.flatten(), auto_encoder.b1.flatten()))
        else:
            for w, b in weights_biases:
                packed = np.concatenate((packed, w.flatten(), b.flatten()))
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
        solution = minimize(min_func_name, min_x0, args=kwargs.values()[0], method='L-BFGS-B', jac=True,
                            options={'maxiter': self.max_iterations, 'disp': True})
        prediction = self.predict(solution.x, X)
        test_accuracy = Y[:, 0] == prediction[:, 0]
        return solution.x, np.mean(test_accuracy)

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
                                            method='L-BFGS-B', options={'maxiter': self.max_iterations, 'disp': True})
            sparse_encoder.unpackWeightsBiases(opt_packed_params.x)
            self.list_auto_encoders.append(sparse_encoder)
            activation = 1. / (1 + np.exp(-(np.dot(sparse_encoder.W1, activation) + sparse_encoder.b1)))
            i += 1

        rand = np.random.RandomState(np.random.seed(int(time())))
        softmax_x0 = 0.005 * np.asarray(rand.normal(size=(self.network_dims[-1]*self.network_dims[-2], 1)))
        softmax_pack, trAcc = self.findAccuracy(tsX, tsY, self.softmaxCost, softmax_x0, accuracy_type='training & softmax',
                                         kwargs=(activation, trY))
        print "Accuracy after training & softmax = {0}".format(trAcc)

        packed_auto_encoder_stack_with_softmax = self.packSoftmaxWeightsAndBiases()
        ftX_5, ftY_5 = self.sample_finetuning_data(trX, trY, samples_per_class=5)
        ftX_1, ftY_1 = self.sample_finetuning_data(ftX_5, ftY_5, samples_per_class=1)
        _, ftAcc1 = self.findAccuracy(tsX, tsY, self.stackedAutoencoderCost, packed_auto_encoder_stack_with_softmax,
                                      kwargs=(ftX_1, ftY_1))
        _, ftAcc5 = self.findAccuracy(tsX, tsY, self.stackedAutoencoderCost, packed_auto_encoder_stack_with_softmax,
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
    trX, trY, tsX, tsY = load_fashion_mnist()
    sae = StackedAutoencoder(max_iterations=100)
    sae.runStackedAutoencoder(trX, trY, tsX, tsY)


if __name__ == "__main__":
    main()