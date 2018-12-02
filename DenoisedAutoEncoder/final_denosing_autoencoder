# Python 3
# Author: Akshay - #1212981859

import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import warnings
import time


def tanh(Z):
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache


def tanh_der(dA, cache):
    # CODE HERE
    A1 = cache["Z"]
    A = 1 - np.square(np.tanh(A1))
    dZ = dA * (1 - A)
    return dZ


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache


def sigmoid_der(dA, cache):
    # CODE
    t = {}
    A, t = sigmoid(cache["Z"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dZ = dA * (A * (1 - A))
    return dZ


def initialize_2layer_weights(n_in, n_h, n_fin):
    # initialize network parameters
    ### CODE HERE
    # print(n_fin, n_in, n_h)
    np.random.seed(0)
    W1 = np.random.randn(n_h, n_in)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_fin, n_h)
    b2 = np.zeros((n_fin, 1))

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters


def linear_forward(A, W, b):
    ### CODE HERE
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    cache["W"] = W
    cache["b"] = b
    return Z, cache


def layer_forward(A_prev, W, b, activation):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache


def cost_estimate(A2, Y):
    ### CODE HERE
    # cost = np.linalg.norm(Y - A2, ord=2)  # take norm of the vector
    #cost = np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    #cost = cost * (-1 / Y.shape[1])
    cost=np.mean(np.square(np.subtract(A2,Y)))
    return cost


def error(A2, Y):
    ### CODE HERE
    # cost = np.linalg.norm(Y - A2, ord=2)  # take norm of the vector
    #cost = np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    #cost = cost * (-1 / Y.shape[1])
    cost=np.mean(np.square(np.subtract(A2,Y)))
    return cost

def linear_backward(dZ, cache, W, b):
    # CODE HERE
    A = cache["A"]
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, activation):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def denoise(X, parameters):
    ### CODE HERE
    A1, cache1 = layer_forward(X, parameters["W1"], parameters["b1"], 'sigmoid')
    YPred, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], 'sigmoid')
    return YPred


def two_layer_network(X, Y, net_dims, num_iterations=2000, learning_rate=0.1):
    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)

    A0 = X
    costs = []
    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE
        A1, cache1 = layer_forward(A0, parameters["W1"], parameters["b1"], 'sigmoid')
        A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], 'sigmoid')

        # cost estimation
        ### CODE HERE
        cost = cost_estimate(A2, Y)

        # Backward Propagation
        ### CODE HERE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            epsilon = 0.000001
            m = Y.shape[1]
            dA2 = 1.0 / m * (np.divide(-1 * Y, A2 + epsilon) + np.divide(1 - Y, 1 - A2 + epsilon))

        dA_prev2, dW2, db2 = layer_backward(dA2, cache2, parameters["W2"], parameters["b2"], 'sigmoid')
        dA_prev1, dW1, db1 = layer_backward(dA_prev2, cache1, parameters["W1"], parameters["b2"], 'sigmoid')

        # update parameters
        ### CODE HERE

        parameters["W1"] = parameters["W1"] - learning_rate * dW1
        parameters["W2"] = parameters["W2"] - learning_rate * dW2
        # parameters["W2"] = parameters["W1"].T
        parameters["b1"] = parameters["b1"] - learning_rate * db1
        parameters["b2"] = parameters["b2"] - learning_rate * db2

        if ii % 10 == 0:
             costs.append(cost)
        if ii % 500 == 0:
            print("Execution at iteration: " + str(ii ) + "!")
            print("cost: " + str(cost) + "!")

    return costs, parameters


def main():
    start_time = time.time()
    train_data, train_label, test_data, test_label = mnist()
    # print(param)
    n_rows = train_data.shape[0]
    n_cols = train_data.shape[1]
    test_rows = test_data.shape[0]
    test_cols = test_label.shape[1]

    mean = 0.0
    stddev = 0.4
    noise = np.random.normal(mean, stddev, (n_rows, n_cols))
    trX_noisy = train_data + noise
    noise = np.random.normal(mean, stddev, (test_rows, test_cols))
    tsX_noisy = test_data + noise

    # fig = plt.figure()
    # plt.imshow(train_data[:, 999].reshape(28, -1))
    # plt.title("Training_Sample")
    # plt.show()
    # fig.savefig("Training_Sample")
    #
    # fig = plt.figure()
    # plt.imshow(trX_noisy[:, 999].reshape(28, -1))
    # plt.title("Noisy_Training_Sample")
    # plt.show()
    # fig.savefig("Noisy_Training_Sample")

    n_in, m = train_data.shape
    n_fin, m = train_data.shape
    n_h = 1000
    net_dims = [n_in, n_h, n_fin]
    # initialize learning rate and num_iterations
    learning_rate = 1
    num_iterations = 1000

    costs, parameters = two_layer_network(trX_noisy, train_data, net_dims, num_iterations=num_iterations,
                                          learning_rate=learning_rate)

    # compute the accuracy for training set and testing set
    # train_Pred = classify(train_data, parameters)

    fig = plt.figure()
    plt.imshow(test_data[:, 99].reshape(28, -1))
    plt.title("Test_Sample")
    plt.show()
    fig.savefig("Test_Sample")

    fig = plt.figure()
    plt.imshow(tsX_noisy[:, 99].reshape(28, -1))
    plt.title("Noisy_Test_Sample")
    plt.show()
    fig.savefig("Noisy_Test_Sample")

    fig = plt.figure()
    test_Pred = denoise(tsX_noisy, parameters)
    print("error of test sample" ,(error(test_Pred,test_data)))
    plt.imshow(test_Pred[:, 99].reshape(28, -1))
    plt.title("Denoised_Test_Sample")
    plt.show()
    fig.savefig("Denoised_Test_Sample")
	
	plt.plot(costs, label = 'Training cost')
    plt.title("training cost with learning rate =" + str(learning_rate)+"iterations"+str(num_iterations)+"noise stddev:"+str(stddev))
    plt.show()
	fig.savefig("training cost")
    print("Total execution time: %s minutes!!" % ((time.time() - start_time)//60))


if __name__ == "__main__":
    main()
