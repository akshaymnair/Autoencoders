# python 3
import matplotlib.pyplot as plt
import sys
from load_mnist import mnist
from sklearn.cluster import KMeans
import numpy as np
import warnings
from sklearn import svm
from sklearn.metrics import accuracy_score

# Save all the Print Statements in a Log file.
print_out = sys.stdout
log_file = open("BaseLine_KMeans_Results.txt", "a")
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


print("Unsupervised Classification on Fashion-MNIST using K-Means algorithm!\n")

n_clusters = len(np.unique(train_label.T))
print("Unique Clusters chosen from input: ", n_clusters)
clf = KMeans(n_clusters=n_clusters)
clf.fit(train_data.T)
y_labels_train = clf.labels_
# print("K means Train labels: ", y_labels_train)
# print("=="*50)
y_labels_test = clf.predict(test_data.T)
# print("K means Test labels: ", y_labels_test)

X_train = y_labels_train[:, np.newaxis]
X_test = y_labels_test[:, np.newaxis]

print('\nFinding accuracy of KMeans using SVM Classifier: ')

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     clf = svm.SVC(gamma=0.1, kernel='poly')
#     clf.fit(X_train, train_label.T)
#
#     acc = clf.score(X_test, test_label.T)
#     print('\nAccuracy of K Means classifier using SVM: ', acc)
#
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    clf = svm.SVC()
    clf.fit(X_train, train_label.T)
    y_pred = clf.predict(X_test)

print('Accuracy: {}'.format(accuracy_score(test_label.T, y_pred)))

print("==" * 50)
print("\n")
sys.stdout = print_out
log_file.close()
