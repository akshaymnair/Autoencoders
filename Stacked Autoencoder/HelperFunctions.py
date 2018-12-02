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
