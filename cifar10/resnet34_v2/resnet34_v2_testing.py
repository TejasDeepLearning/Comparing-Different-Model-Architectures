import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Add, Activation, AveragePooling2D, ZeroPadding2D, PReLU
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from keras.initializers import glorot_uniform, Constant

# check if tf is using gpu
print(tf.config.list_physical_devices('GPU'))

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

K = len(set(ytrain))
print(f'number of classes = {K}')

model = tf.keras.models.load_model("/home/tejas/Documents/Deep Learning/My Work/CNN/cifar10/resnet34/resnet34_v2.h5")

print(model.evaluate(xtest, ytest))