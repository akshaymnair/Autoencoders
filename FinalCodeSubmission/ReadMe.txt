* How to run the Denoising AutoEncoder
1. load Fashion MNIST data sets into the same directory as the denoising autoencoder files
>>python denoisewithgaussiannoise.py // for gaussian noises
>>python denoisewithrandomnoise.py // for random noises

the parameters like Learning rates and Noise rates can be changed in the code

* How to run the Stacked Autoencoder
1. Load Fashion MNIST data in to the Stacked autoencoder folder
>>python StackedAutoencoder.py -e 200 -l "[300, 200, 100]"
here e is epochs
and [x ,y ,z ] are the hidden layer size

other parameters can be changed from the code

* How to run the Baseline classifiers
1. Load the Fashion MNIST into the Baseline folder
>>python Baseline_Stacked_Kmeans.py
>>python Baseline_Stacked_LogicRegression.py
>>python Baseline_Stacked_SVM.py
>>python BaseLine_Stacked_NeuralNet.py