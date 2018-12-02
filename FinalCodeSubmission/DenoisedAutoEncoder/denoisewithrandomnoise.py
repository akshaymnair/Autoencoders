# Python 2.7

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
    # cost = np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    # cost = cost * (-1 / Y.shape[1])
    cost = np.mean(np.square(np.subtract(A2, Y)))
    return cost


def error(A2, Y):
    ### CODE HERE
    # cost = np.linalg.norm(Y - A2, ord=2)  # take norm of the vector
    # cost = np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    # cost = cost * (-1 / Y.shape[1])
    cost = np.mean(np.square(np.subtract(A2, Y)))
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
            epsilon = 0.00000001
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
            print("Execution at iteration: " + str(ii) + "!")
            print("cost: " + str(cost) + "!")

    return costs, parameters


def masking_noise(X, v):
    noisy_X = X.copy()
    n_samples = X.shape[0]
    n_features = X.shape[1]
    for i in range(n_samples):
        mask = np.random.randint(0, n_features, v)
        for m in mask:
            noisy_X[i][m] = 0.
    return noisy_X


def get_corrupted_input(input, stddev):
    noise = np.random.normal(0, stddev, (input.shape[0], input.shape[1]))
    return (input + noise * 0.2)


def main():
    start_time = time.time()
    train_data, train_label, test_data, test_label = mnist()
    # print(param)
    n_rows = train_data.shape[0]
    n_cols = train_data.shape[1]
    test_rows = test_data.shape[0]
    test_cols = test_label.shape[1]


    noise = 5
    testnoise1 = 4
    testnoise2 = 5
    testnoise3 = 6
    testnoise4 = 7
    trX_noisy = masking_noise(train_data, noise)
    tsX_noisy1 = masking_noise(test_data, testnoise1)
    tsX_noisy2 = masking_noise(test_data, testnoise2)
    tsX_noisy3 = masking_noise(test_data, testnoise3)
    tsX_noisy4 = masking_noise(test_data, testnoise4)



    n_in, m = train_data.shape
    n_fin, m = train_data.shape
    n_h = 1000
    net_dims = [n_in, n_h, n_fin]
    # initialize learning rate and num_iterations
    learning_rate = 1
    num_iterations = 1000

    costs, parameters = two_layer_network(trX_noisy, train_data, net_dims, num_iterations=num_iterations,
                                          learning_rate=learning_rate)



    fig = plt.figure()
    plt.imshow(test_data[:, 99].reshape(28, -1), cmap="gray")
    plt.title("Test_Sample  with random noise"+str(noise)+"is")
    plt.show()
    fig.savefig("Test_Sample with random noise"+str(noise)+"is.png")

    fig = plt.figure()
    plt.imshow(test_data[:, 72].reshape(28, -1), cmap="gray")
    plt.title("Test_Sample with random noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Test_Sample 8with random noise" + str(noise) + "is.png")
    fig = plt.figure()
    plt.imshow(test_data[:, 20].reshape(28, -1), cmap="gray")
    plt.title("Test_Sample with random noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Test_Sample 9 with random noise" + str(noise) + "is.png")

    fig = plt.figure()
    plt.imshow(tsX_noisy1[:, 99].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample..with_random_noise"+str(testnoise1)+"and train noise"+str(noise)+"is")
    plt.show()
    fig.savefig("Noisy_Test_Samplewith_random_noise"+str(testnoise1)+"and train noise"+str(noise)+"is.png")
    fig = plt.figure()
    plt.imshow(tsX_noisy2[:, 99].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample with_random_noise"+str(testnoise2)+"and train noise"+str(noise)+"is")
    plt.show()
    fig.savefig("Noisy_Test_Sample with_random_noise"+str(testnoise2)+"and train noise"+str(noise)+"is.png")
    fig = plt.figure()
    plt.imshow(tsX_noisy3[:, 99].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample with_random_noise"+str(testnoise3)+"and train noise"+str(noise)+"is")
    plt.show()
    fig.savefig("Noisy_Test_Sample with_random_noise"+str(testnoise3)+"and train noise"+str(noise)+"is.png")

    fig = plt.figure()
    plt.imshow(tsX_noisy4[:, 99].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample with_random_noise"+str(testnoise4)+"and train noise"+str(noise)+"is")
    plt.show()
    fig.savefig("Noisy_Test_Sample with_random_noise"+str(testnoise4)+"and train noise"+str(noise)+"is.png")
    fig = plt.figure()

    fig = plt.figure()
    plt.imshow(tsX_noisy1[:, 72].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample 8with_random_noise" + str(testnoise1) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Noisy_Test_Sample8with_random_noise" + str(testnoise1) + "and train noise" + str(noise) + "is.png")
    fig = plt.figure()
    plt.imshow(tsX_noisy2[:, 72].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample 8 with_random_noise" + str(testnoise2) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Noisy_Test_Sample8 with_random_noise" + str(testnoise2) + "and train noise" + str(noise) + "is.png")
    fig = plt.figure()
    plt.imshow(tsX_noisy3[:, 72].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample 8with_random_noise" + str(testnoise3) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Noisy_Test_Sample 8with_random_noise" + str(testnoise3) + "and train noise" + str(noise) + "is.png")

    fig = plt.figure()
    plt.imshow(tsX_noisy4[:, 72].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample 8with_random_noise" + str(testnoise4) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Noisy_Test_Sample8 with_random_noise" + str(testnoise4) + "and train noise" + str(noise) + "is.png")
    fig = plt.figure()

    fig = plt.figure()
    plt.imshow(tsX_noisy1[:, 20].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample 9with_random_noise" + str(testnoise1) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Noisy_Test_Sample8with_random_noise" + str(testnoise1) + "and train noise" + str(noise) + "is.png")
    fig = plt.figure()
    plt.imshow(tsX_noisy2[:, 20].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample 9 with_random_noise" + str(testnoise2) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Noisy_Test_Sample9 with_random_noise" + str(testnoise2) + "and train noise" + str(noise) + "is.png")
    fig = plt.figure()
    plt.imshow(tsX_noisy3[:, 20].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample 9with_random_noise" + str(testnoise3) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Noisy_Test_Sample 9 with_random_noise" + str(testnoise3) + "and train noise" + str(noise) + "is.png")

    fig = plt.figure()
    plt.imshow(tsX_noisy4[:, 20].reshape(28, -1), cmap="gray")
    plt.title("Noisy_Test_Sample 9 with_random_noise" + str(testnoise4) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Noisy_Test_Sample 9 with_random_noise" + str(testnoise4) + "and train noise" + str(noise) + "is.png")




    test_Pred1 = denoise(tsX_noisy1, parameters)
    test_Pred2 = denoise(tsX_noisy2, parameters)
    test_Pred3 = denoise(tsX_noisy3, parameters)
    test_Pred4 = denoise(tsX_noisy4, parameters)
    print("accuracy of test sample 2 with  random noise" + str(testnoise1) + "is  " + str(
        (1 - error(test_Pred1, test_data)) * 100) + "%")
    print("accuracy of test sample 2 with random  noise" + str(testnoise2) + "is  " + str(
        (1 - error(test_Pred2, test_data)) * 100) + "%")
    print("accuracy of test sample 2 with random  noise" + str(testnoise3) + "is  " + str(
        (1 - error(test_Pred3, test_data)) * 100) + "%")
    print("accuracy of test sample 2 with random noise" + str(testnoise4) + "is  " + str(
        (1 - error(test_Pred4, test_data)) * 100) + "%")
    fig = plt.figure()
    plt.imshow(test_Pred1[:, 99].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample with_random_noise"+str(testnoise1)+"and train noise"+str(noise)+"is")
    plt.show()

    fig.savefig("Denoised_random noise_random_Test_Sample op1")
    fig = plt.figure()
    plt.imshow(test_Pred2[:, 99].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample with_random_noise"+str(testnoise2)+"and train noise"+str(noise)+"is")
    plt.show()
    fig.savefig("Denoised_random noiseTest_Sample op2")
    fig = plt.figure()
    plt.imshow(test_Pred3[:, 99].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample with_random_noise"+str(testnoise3)+"and train noise"+str(noise)+"is")
    plt.show()
    fig.savefig("Denoised_random_noiseTest_Sample op3")
    fig = plt.figure()
    plt.imshow(test_Pred4[:, 99].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample with_random_noise"+str(testnoise4)+"and train noise"+str(noise)+"is")
    plt.show()
    fig.savefig("Denoised_random noiseTest_Sample op4")

    fig = plt.figure()
    plt.imshow(test_Pred1[:, 72].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample 8 with_random_noise" + str(testnoise1) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Denoised_random noise_random_Test_Sample8 op1")
    fig = plt.figure()
    plt.imshow(test_Pred2[:, 72].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample with_random_noise" + str(testnoise2) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Denoised_random noiseTest_Sample 8 op2")
    fig = plt.figure()
    plt.imshow(test_Pred3[:, 72].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample with_random_noise" + str(testnoise3) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Denoised_random_noiseTest_Sample 8 op3")
    fig = plt.figure()
    plt.imshow(test_Pred4[:, 72].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample with_random_noise" + str(testnoise4) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Denoised_random noiseTest_Sample 8 op4")



    fig = plt.figure()
    plt.imshow(test_Pred1[:, 20].reshape(28, -1), "gray")
    plt.title("denoised_Sample 9 with_random_noise" + str(testnoise1) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Denoised_random noise_random_Test_Sample9 op1")
    fig = plt.figure()
    plt.imshow(test_Pred2[:, 20].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample 9 with_random_noise" + str(testnoise2) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Denoised_random noiseTest_Sample 9 op2")
    fig = plt.figure()
    plt.imshow(test_Pred3[:, 20].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample 9 with_random_noise" + str(testnoise3) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Denoised_random_noiseTest_Sample 9 op3")
    fig = plt.figure()
    plt.imshow(test_Pred4[:, 20].reshape(28, -1), cmap="gray")
    plt.title("denoised_Sample 9 with_random_noise" + str(testnoise4) + "and train noise" + str(noise) + "is")
    plt.show()
    fig.savefig("Denoised_random noiseTest_Sample 9 op4")





    plt.plot(costs, label='Training cost')

    plt.title("training cost with learning rate =" + str(learning_rate) + "iterations" + str(
        num_iterations) + "noise :" + str(noise))
    plt.show()
    fig.savefig("trainingCost")
    print("Total execution time: %s minutes!!" % ((time.time() - start_time) // 60))


if __name__ == "__main__":
    main()
