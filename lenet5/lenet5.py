import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

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

# set the number of classes
K = len(set(ytrain))
print(f'number of classes = {K}')

# build the model (LeNet5)
i = Input(xtrain[0].shape)
x = Conv2D(filters=6, kernel_size=(5, 5), activation='relu')(i)
x = BatchNormalization()(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
x = BatchNormalization()(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Conv2D(filters=120, kernel_size=(5, 5), activation='relu')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(units=84, activation='relu')(x)
x = Dense(units=K, activation='softmax')(x)

model = Model(i, x)
print(model.summary())

# compile the model
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# save the best model using a callback 
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='lenet5.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# data augmentation
batch_size = 512
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    featurewise_center=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=24
)
train_generator = data_generator.flow(xtrain, ytrain, batch_size)
steps_per_epoch = xtrain.shape[0] // batch_size

# fit and train the model 
r = model.fit(
    train_generator,
    validation_data=(xtest, ytest),
    steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint],
    epochs=50
)

# plot accuracy
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# plot loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()