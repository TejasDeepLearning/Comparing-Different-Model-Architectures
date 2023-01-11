import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

model = tf.keras.models.load_model("/home/tejas/Documents/Deep Learning/My Work/CNN/cifar10/vgg19/vgg19.h5")
tf.keras.utils.plot_model(model, to_file='/home/tejas/Documents/Deep Learning/My Work/CNN/cifar10/vgg19/vgg19.png', show_shapes=True, show_layer_names=True, show_layer_activations=True, dpi=300)

# load in the data 
cifar10 = tf.keras.datasets.cifar10 
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
print(f'shapes of everything: {xtrain.shape, ytrain.shape, xtest.shape, ytest.shape}')

# normalize the inputs
minmax = MinMaxScaler()
xtrain = minmax.fit_transform(xtrain.reshape(50000, 3072), ytrain)
xtest = minmax.fit_transform(xtest.reshape(10000, 3072), ytest)
# reshape it again after normalization 
xtrain = xtrain.reshape(50000, 32, 32, 3)
xtest = xtest.reshape(10000, 32, 32, 3)

# flatten the targets
ytrain, ytest = ytrain.flatten(), ytest.flatten()

print(model.evaluate(xtest, ytest))