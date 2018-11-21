import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

datasets_dir = 'C:\\Users\\amit1\\Documents\\python-workspace\\FSL Assignment 3\\'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    data_dir = os.path.join(datasets_dir, 'mnist')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY


def main():
    trX, trY, tsX, tsY = mnist(noTrSamples=2000,
                               noTsSamples=1000, digit_range=[2, 7],
                               noTrPerClass=1000, noTsPerClass=500)

    plt.imshow(trX[:, 1500].reshape(28, -1))
    plt.show()
    trY[0, 10]


    # creating the noisy test data by adding X_test with noise
    n_rows = trX.shape[0]
    n_cols = trX.shape[1]
    mean = 0.0
    stddev = 1.0
    factor = 0.1
    while factor <= 1.0:
        noise = factor * np.random.normal(mean, stddev, (n_rows, n_cols))
        X_test_noisy = trX + noise

        plt.imshow(X_test_noisy[:,1500].reshape(28, -1))
        plt.show()
        trY[0, 10]
        # stddev += 0.1
        factor += 0.1


def separate_train_val_data(noTotalTrainVal, noTrPerClass, noValPerClass, train_val_data, train_val_label):
    assert (noTotalTrainVal%(noTrPerClass + noValPerClass) == 0), 'data class division mismatch'
    i, train_data, train_label, val_data, val_label = 0, None, None, None, None
    while i < noTotalTrainVal:
        itr_train_data = train_val_data[:, i:i + noTrPerClass]
        itr_train_label = train_val_label[:, i:i + noTrPerClass]
        itr_val_data = train_val_data[:, i + noTrPerClass:i +noTrPerClass + noValPerClass]
        itr_val_label = train_val_label[:, i + noTrPerClass:i + noTrPerClass + noValPerClass]
        if train_data is None:
            train_data, train_label, val_data, val_label = itr_train_data, itr_train_label, itr_val_data, itr_val_label
        else:
            train_data = np.append(train_data, itr_train_data, axis=1)
            train_label = np.append(train_label, itr_train_label, axis=1)
            val_data = np.append(val_data, itr_val_data, axis=1)
            val_label = np.append(val_label, itr_val_label, axis=1)
        i += noTrPerClass + noValPerClass
    return train_data, train_label, val_data, val_label
    
if __name__ == "__main__":
    main()
