import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras import backend as K
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler

from keras.utils.np_utils import to_categorical

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
# import keras
import numpy as np
from scipy.linalg import eigh

from skimage.transform import resize

# def preprocess(array):
#     """
#     Normalizes the supplied array and reshapes it into the appropriate format.
#     """

#     array = array.astype("float32") / 255.0
#     tmpArr = np.zeros([len(array), 224, 224])
#     for i in range(len(array)):
#         tmpArr[i] = (resize(array[i],(224, 224)))

#     return tmpArr

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    # tmpArr = np.zeros([len(array), 224, 224])
    # for i in range(len(array)):
    #     tmpArr[i] = (resize(array[i],(224, 224)))

    return array


def resizeArr(array):
    # array = array.astype("float32")
    tmpArr = np.zeros([len(array), 224, 224])
    for i in range(len(array)):
        tmpArr[i] = (resize(array[i],(224, 224)))
    
    return tmpArr

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()


x_train = x_train[-20000:]
y_train = y_train[-20000:]

x_test = x_train[-2000:]
y_test = y_train[-2000:]

x_train = preprocess(x_train)
x_test = preprocess(x_test)

X_train = x_train.reshape(-1, 784)
X_test = x_test.reshape(-1, 784)

pca = PCA(196)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# std_scaler = StandardScaler()
# X_train_pca = std_scaler.fit_transform(X_train_pca)
# X_test_pca = std_scaler.transform(X_test_pca)
# x_train = std_scaler.fit_transform(X_train)
# x_test = std_scaler.transform(X_test)


# x_train = X_train_pca.reshape(X_train_pca.shape[0], 224, 224)
# x_test = X_test_pca.reshape(X_test_pca.shape[0], 224, 224)
x_train = resizeArr(X_train_pca)
x_test = resizeArr(X_test_pca)


# print(x_train.shape)
# quit()

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_train = tf.cast(x_train, dtype=tf.float16)
x_test = tf.cast(x_test, dtype=tf.float16)

x_train = tf.image.grayscale_to_rgb(x_train)
x_test = tf.image.grayscale_to_rgb(x_test)

SAVE_PATH = "pca/" # Data saving folder

pca_train_images = np.asarray(x_train)
pca_test_images = np.asarray(x_test)

np.save(SAVE_PATH + "pca_train_images.npy", pca_train_images)
np.save(SAVE_PATH + "pca_test_images.npy", pca_test_images)

x_val = x_train[-2000:,:,:,:]
y_val = y_train[-2000:]
x_train = x_train[:-2000,:,:,:]
y_train = y_train[:-2000]

# print(x_train.shape)
# quit()

model = models.Sequential()
model.add(layers.Input(shape=x_train.shape[1:]))


model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))

model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))

model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_val, y_val))

fig, axs = plt.subplots(2, 1, figsize=(15,15))

axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train', 'Val'])

axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train', 'Val'])

plt.show()

model.save_weights("keras_models/pca_model")

model.evaluate(x_test, y_test)