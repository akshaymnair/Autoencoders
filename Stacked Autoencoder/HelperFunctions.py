import array
import os
import struct
import numpy as np
from scipy.sparse import csr_matrix


mnist_dir = 'fashion-mnist'
figures_dir = 'figures'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loadFashionMNISTImages(file_name):
    file_path = os.path.join(mnist_dir, file_name)
    f = open(file_path, 'rb')
    f.read(4)
    num_images = struct.unpack('>I', f.read(4))[0]
    num_rows = struct.unpack('>I', f.read(4))[0]
    num_cols = struct.unpack('>I', f.read(4))[0]
    dataset = np.zeros((num_rows * num_cols, num_images))
    images_raw = array.array('B', f.read())
    f.close()
    each_row_len = num_rows * num_cols
    for i in range(num_images):
        dataset[:, i] = images_raw[each_row_len * i: each_row_len * (i + 1)]
    return dataset / 255


def loadFashionMNISTLabels(file_name):
    file_path = os.path.join(mnist_dir, file_name)
    f = open(file_path, 'rb')
    f.read(4)
    num_examples = struct.unpack('>I', f.read(4))[0]
    labels = np.zeros((num_examples, 1), dtype=np.int)
    labels_raw = array.array('b', f.read())
    f.close()
    labels[:, 0] = labels_raw[:]
    return labels


def loadFashionMNIST():
    trX = loadFashionMNISTImages('train-images-idx3-ubyte')
    trY = loadFashionMNISTLabels('train-labels-idx1-ubyte')
    tsX = loadFashionMNISTImages('t10k-images-idx3-ubyte')
    tsY = loadFashionMNISTLabels('t10k-labels-idx1-ubyte')
    return trX, trY, tsX, tsY


def filterMNIST(X, Y, samples_per_class=1, num_classes=10):
    d = {}
    data, label = [], []
    for i in range(X.shape[-1]):
        digit = Y[i, 0]
        if not digit in d:
            d[digit] = []
        if len(d[digit]) < samples_per_class:
            d[digit].append(i)
            data.append(X[:, i])
            label.append(Y[i])
        if len(data) == samples_per_class * num_classes:
            break
    data, label = np.array(data).T, np.array(label)
    return data, label


def getGroundTruth(Y):
    Y = np.array(Y).flatten()
    ones = np.ones(len(Y))
    indexes = np.arange(len(Y) + 1)
    ground_truth = csr_matrix((ones, Y, indexes)).todense().T
    return ground_truth


def separate_train_val_data(noTotalTrainVal, noTrPerClass, noValPerClass, numClasses, train_val_data, train_val_label):
    assert (noTotalTrainVal == (noTrPerClass + noValPerClass) * numClasses), 'data class division mismatch'
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


def shuffleDataLabels(X, Y):
    s = np.arange(X.shape[1])
    np.random.shuffle(s)
    X = X[:, s]
    Y = (Y[:, s]).T
    return X, Y


def sample_finetuning_data(trX, trY, no_of_classes, samples_per_class=1):
    total_data_size = trX.shape[-1]
    no_val_per_class = (total_data_size - (no_of_classes * samples_per_class)) / no_of_classes
    ftX, ftY, _, _ = separate_train_val_data(trX.shape[1], samples_per_class, no_val_per_class, no_of_classes, trX, trY)
    ftX, ftY = shuffleDataLabels(ftX, ftY)
    return ftX, ftY