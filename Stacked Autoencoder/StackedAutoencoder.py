import argparse
import HelperFunctions as helper
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import minimize
from SparseAutoencoder import SparseAutoencoder
from time import time


class StackedAutoencoder:
    def __init__(self, input_size=784, hidden_layer_sizes=[500, 250, 100], output_size=10, max_iterations=1000, method='L-BFGS-B'):
        self.max_iterations = max_iterations  # number of optimization iterations
        self.method = method    # method for optimization
        self.rho = 0.1          # average activation of hidden units
        self.lamda = 0.003      # weight decay param
        self.beta = 3           # sparsity penalty term weight

        self.network_dims = [input_size] + hidden_layer_sizes + [output_size]
        self.list_auto_encoders = []
        self.packed_auto_encoder_stack_with_softmax = []

    def softmaxCost(self, params, X, labels):

        params = params.reshape(self.network_dims[-1], self.network_dims[-2])
        hypothesis = np.exp(np.dot(params, X))
        probs = hypothesis / np.sum(hypothesis, axis=0)

        ground_truth = helper.getGroundTruth(labels)
        weight_decay = 0.5 * self.lamda * np.sum(np.multiply(params, params))
        cost = np.multiply(ground_truth, np.log(probs))
        cost = -(np.sum(cost) / X.shape[1]) + weight_decay

        gradient = -np.dot(ground_truth - probs, X.T)
        gradient = gradient / X.shape[1] + self.lamda * params
        gradient = np.array(gradient).flatten()

        return [cost, gradient]

    def predict(self, X, params):
        classifier_length = self.network_dims[-1] * self.network_dims[-2]
        classifier = params[0: classifier_length].reshape(self.network_dims[-1], self.network_dims[-2])

        activation = X
        start = classifier_length
        for autoencoder in self.list_auto_encoders:
            W_limits = (start, start + autoencoder.hidden_size * autoencoder.input_size)
            b_limits = (W_limits[1], W_limits[1] + autoencoder.hidden_size)
            W = params[W_limits[0]: W_limits[1]].reshape(autoencoder.hidden_size, autoencoder.input_size)
            b = params[b_limits[0]: b_limits[1]].reshape(autoencoder.hidden_size, 1)
            start = b_limits[1]
            activation = helper.sigmoid(np.dot(W, activation) + b)

        hypothesis = np.exp(np.dot(classifier, activation))
        probs = hypothesis / np.sum(hypothesis, axis=0)
        predictions = np.zeros((X.shape[1], 1))
        predictions[:, 0] = np.argmax(probs, axis=0)
        return predictions

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

    def fineTune(self, ftX, ftY, tsX, tsY, spc):
        print "Fine Tuning Phase - {0} samples per class(SPC): In progress".format(str(spc))
        fine_tuned_solution = minimize(self.fineTuneFit, self.packed_auto_encoder_stack_with_softmax,
                                    args=(ftX, ftY,), method=self.method, jac=True,
                                    options={'maxiter': self.max_iterations, 'disp': True})
        fine_tuned_solution = fine_tuned_solution.x
        predictions = self.predict(tsX, fine_tuned_solution)
        correct = tsY[:, 0] == predictions[:, 0]
        print "Fine Tuning Phase - {0} samples per class(SPC): Completed".format(spc)
        return np.mean(correct)

    def fineTuneFit(self, packed_stack_with_softmax, X, Y):
        classifier, unpacked_wb = self.unpackSoftmaxWeightsAndBiases(packed_stack_with_softmax)

        layers_count = len(unpacked_wb)
        activation = {0: X}
        for i in range(layers_count):
            activation[i + 1] = helper.sigmoid(np.dot(unpacked_wb[i][0], activation[i]) + unpacked_wb[i][1])

        ground_truth = helper.getGroundTruth(Y)
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

    def fit(self, trX, trY, tsX, tsY):
        print "Training Phase - Model fitting: In progress"
        i, features = 1, trX
        while i < len(self.network_dims) - 1:
            sparse_encoder = SparseAutoencoder(self.network_dims[i-1], self.network_dims[i], self.rho, self.lamda, self.beta)
            packed_inputs = sparse_encoder.packWeightsBiases()
            opt_packed_params = minimize(fun=sparse_encoder.fit, x0=packed_inputs, args=(features, ), jac=True,
                                            method=self.method, options={'maxiter': self.max_iterations, 'disp': True})
            sparse_encoder.features = opt_packed_params.x
            sparse_encoder.unpackWeightsBiases(opt_packed_params.x)
            self.list_auto_encoders.append(sparse_encoder)
            features = 1. / (1 + np.exp(-(np.dot(sparse_encoder.W1, features) + sparse_encoder.b1)))
            i += 1

        rand = np.random.RandomState(np.random.seed(int(time())))
        softmax_x0 = 0.005 * np.asarray(rand.normal(size=(self.network_dims[-1]*self.network_dims[-2], 1)))
        softmax_solution = minimize(self.softmaxCost, softmax_x0, args=(features, trY,), method=self.method, jac=True,
                            options={'maxiter': self.max_iterations, 'disp': True})
        softmax_solution = softmax_solution.x
        self.packed_auto_encoder_stack_with_softmax = self.packSoftmaxWeightsAndBiases(softmax_solution)
        predictions = self.predict(tsX, self.packed_auto_encoder_stack_with_softmax)
        correct = tsY[:, 0] == predictions[:, 0]
        trAcc = np.mean(correct)
        print "Training Phase - Model fitting: Completed"
        return trAcc

    def plotGraph(self, title):
        for i in range(len(self.list_auto_encoders)):
            plt.plot(self.list_auto_encoders[i].costs, label='Layer - %i' % (i + 1))
        plt.xlabel('Iterations -->')
        plt.ylabel('Training Cost -->')
        plt.title(title)
        plt.xscale('log')
        plt.legend(title='Sparse Autoencoder', fancybox=True, fontsize='x-small')
        if not os.path.exists(helper.figures_dir):
            os.makedirs(helper.figures_dir)
        plt.savefig(os.path.join(helper.figures_dir, title + '.png'))
        plt.show()


def SDA(show_cost_graph=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--hidden', type=str, default="[500, 200, 100]")
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-m', '--method', type=str, default='L-BFGS-B')
    parser.add_argument('-i', '--input', type=int, default=784)
    parser.add_argument('-o', '--output', type=int, default=10)
    args = parser.parse_args()
    hidden_layers = [int(h) for h in args.hidden.strip('[').strip(']').split(',' if ',' in args.hidden else ' ')]

    from load_fashion_mnist import load_fashion_mnist
    trX1, trY1, tsX1, tsY1 = load_fashion_mnist()
    trainX, trainY = helper.shuffleDataLabels(trX1, trY1)
    testX, testY = helper.shuffleDataLabels(tsX1, tsY1)
    ftX_51, ftY_51 = helper.sample_finetuning_data(trX1, trY1, args.output, samples_per_class=5)
    ftX_11, ftY_11 = helper.sample_finetuning_data(trX1, trY1, args.output, samples_per_class=1)

    # sae1 = StackedAutoencoder(input_size=args.input, hidden_layer_sizes=hidden_layers, output_size=args.output,
    #                          max_iterations=args.epochs, method=args.method)
    # sae1.runStackedAutoencoder(trainX, trainY, testX, testY)
    # ftAcc_1 = sae1.fineTune(trainX, trainY, testX, testY)
    # ftAcc5_1 = sae1.fineTune(ftX_51, ftY_51, tsX1, tsY1)
    # ftAcc1_1 = sae1.fineTune(ftX_11, ftY_11, tsX1, tsY1)
    # print "Accuracy after fine-tuning with 1-labeled samples per class = {0}".format(ftAcc1_1)
    # print "Accuracy after fine-tuning with 5-labeled samples per class = {0}".format(ftAcc5_1)
    # print "Accuracy after fine-tuning with train data as labeled samples = {0}".format(ftAcc_1)


    trX, trY, tsX, tsY = helper.loadFashionMNIST()
    ftX_5, ftY_5 = helper.filterMNIST(trX, trY, 5)
    ftX_1, ftY_1 = helper.filterMNIST(trX, trY, 1)
    sae = StackedAutoencoder(input_size=args.input, hidden_layer_sizes=hidden_layers, output_size=args.output,
                              max_iterations=args.epochs, method=args.method)
    trAcc = sae.fit(trX, trY, tsX, tsY)
    ftAcc1 = sae.fineTune(ftX_1, ftY_1, tsX, tsY, spc=1)
    ftAcc5 = sae.fineTune(ftX_5, ftY_5, tsX, tsY, spc=5)
    ftAcc = sae.fineTune(trX, trY, tsX, tsY, spc='all')
    print "Accuracy after training & softmax for {0} epochs on {1} hidden layers = {2}".format(args.epochs, str(args.hidden), trAcc)
    print "Accuracy after fine-tuning with 1-labeled SPC for {0} epochs on {1} hidden layers = {2}".format(args.epochs, str(args.hidden), ftAcc1)
    print "Accuracy after fine-tuning with 5-labeled SPC for {0} epochs on {1} hidden layers = {2}".format(args.epochs, str(args.hidden), ftAcc5)
    print "Accuracy after fine-tuning with fullly labeled SPC  for {0} epochs on {1} hidden layers = {2}".format(args.epochs, str(args.hidden), ftAcc)
    if show_cost_graph:
        title = 'Training Cost vs Iteration - {0} epochs and {1} hidden layers'.format(args.epochs, str(args.hidden))
        sae.plotGraph(title)
    return trAcc, ftAcc1, ftAcc5, ftAcc


if __name__ == "__main__":
    SDA()
