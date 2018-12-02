
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

datasets_dir = './'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=60000, noTsSamples=100, digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], noTrPerClass=6000,
          noTsPerClass=10):
    assert noTrSamples == noTrPerClass * len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples == noTsPerClass * len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte.dms'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte.dms'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte.dms'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.dms'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData / 255.
    tsData = tsData / 255.

    tsX = np.zeros((noTsSamples, 28 * 28))
    trX = np.zeros((noTrSamples, 28 * 28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count * noTrPerClass, (count + 1) * noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count * noTsPerClass, (count + 1) * noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1

    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx, :]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY


def show(image, label):
    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = 10
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(image[:, i].reshape(28, -1))
        print(label[:,i])
    plt.show()


# def process_mnist():
#     trX, trY, tsX1, tsY1 = mnist(noTrSamples=6000,
#                                  noTsSamples=100, digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                                  noTrPerClass=600, noTsPerClass=100)
#     tsX = tsX1[:, :1000]
#     tsY = tsY1[:, :1000]
#     vaX = tsX1[:, 1000:]  # validation data set
#     vaY = tsY1[:, 1000:]  # validation labels
#
#     return trX, trY, tsX, tsY, vaX, vaY


def main():
    trX, trY, tsX1, tsY1 = mnist(noTrSamples=60000,
                                 noTsSamples=1400, digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 noTrPerClass=6000, noTsPerClass=700)

    trX, trY, tsX, tsY, vaX, vaY = mnist()

    print("Validation sets: ", vaY.shape, vaX.shape)
    print("Test sets: ", tsY.shape, tsX.shape)
    print("Train sets: ", trY.shape, trX.shape)

    show(vaX,vaY)
    # plt.imshow(vaX[:, 4].reshape(28, -1))
    # plt.show()
    # print(vaY[0, 4])


if __name__ == "__main__":
    main()
