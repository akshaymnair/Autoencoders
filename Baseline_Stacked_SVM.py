# python 3
import matplotlib.pyplot as plt
import sys
from load_mnist import mnist
from sklearn import svm
import warnings

# Save all the Print Statements in a Log file.
print_out = sys.stdout
log_file = open("BaseLine_SVM_Results.txt", "a")
sys.stdout = log_file

train_data, train_label, test_data, test_label = mnist(noTrSamples=60000, noTsSamples=10000,
                                                       digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                       noTrPerClass=6000, noTsPerClass=1000)

# print(train_data.shape, test_data.shape)
# plt.figure()
# plt.imshow(train_data[:, 0].reshape(28, -1), cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    clf = svm.SVC(gamma=0.1, kernel='poly')
    clf.fit(train_data.T, train_label.T)

    acc = clf.score(test_data.T, test_label.T)
    print('\nAccuracy of SVM Classifier: ', acc)

print("==" * 50)
print("\n")
sys.stdout = print_out
log_file.close()
