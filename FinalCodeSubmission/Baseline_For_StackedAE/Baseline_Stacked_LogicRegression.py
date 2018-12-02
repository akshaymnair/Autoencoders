# python 3
import matplotlib.pyplot as plt
import sys
from load_mnist import mnist
from sklearn.linear_model import LogisticRegression
import warnings

# Save all the Print Statements in a Log file.
print_out = sys.stdout
log_file = open("BaseLine_Regression_Results.txt", "a")
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(train_data.T, train_label.T)
    score = logisticRegr.score(test_data.T, test_label.T)

print("\nLogical Regression classifier with Solver = 'lgfgs'")
print("\nAccuracy of Logical Regression classifier: ", score)
print("==" * 50)
print("\n")
sys.stdout = print_out
log_file.close()
