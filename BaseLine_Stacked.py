# python 3

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from load_mnist import mnist

train_data, train_label, test_data, test_label = mnist(noTrSamples=60000, noTsSamples=10000,
                                                       digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                       noTrPerClass=6000, noTsPerClass=1000)

print(train_data.shape, test_data.shape)

# plt.figure()
# plt.imshow(train_data[:, 0].reshape(28, -1), cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

fig = plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[:, i].reshape(28, -1), cmap=plt.cm.binary)
# plt.show()
fig.savefig("InputImages_GreyScale")

l1, l2, l3 = 500, 200, 100
act = 'sigmoid'

model = keras.Sequential([
    keras.layers.Dense(l1, activation=tf.nn.sigmoid),
    keras.layers.Dense(l2, activation=tf.nn.sigmoid),
    keras.layers.Dense(l3, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data.T, train_label.T, epochs=5)

test_loss, test_acc = model.evaluate(test_data.T, test_label.T)

print('Test accuracy:', test_acc)
# print('Test loss:', test_loss)

with open('BaseLine_NeuralNet.txt', 'a') as file:
    file.write('\nTesting Parameters \n')
    file.write('\nHidden Layer Sizes: ' + str(l1) + ' >> ' + str(l2) + ' >> ' + str(l3))
    file.write('\nActivations: ' + act)
    file.write('\nTest Accuracy: ' + str(test_acc))
    file.write('\n')
    file.write('==' * 50)

