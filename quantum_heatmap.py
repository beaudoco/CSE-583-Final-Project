import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
from tensorflow import keras

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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.linalg import eigh

from skimage.transform import resize

import seaborn as sns

#get labels
(_, y_train), (_, y_test)=tf.keras.datasets.mnist.load_data()
y_train = y_train[-20000:]
y_test = y_train[-2000:]

SAVE_PATH = "quanvolution/" # Data saving folder
# Load pre-processed images
q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
q_test_images = np.load(SAVE_PATH + "q_test_images.npy")

#distribute between test and validate
x_val = q_train_images[-2000:,:,:,:]
y_val = y_train[-2000:]
x_train = q_train_images[:-2000,:,:,:]
y_train = y_train[:-2000]

#create same model
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

#load saved model
model.load_weights("keras_models/quantum_model")

#begin creating confusion for training
train_prediction = model.predict(x_train)
val_prediction = model.predict(x_val)
test_prediction = model.predict(q_test_images)

cf_train_matrix = confusion_matrix(y_train, train_prediction.argmax(axis=-1))

cf_train_percent = np.zeros([len(cf_train_matrix), len(cf_train_matrix[0])])
for idx in range(len(cf_train_matrix)):
    cf_train_percent[idx] = cf_train_matrix[idx] / np.sum(cf_train_matrix[idx])

# sns.heatmap(cf_train_matrix, annot=True, )
ax_train = sns.heatmap(cf_train_percent, annot=True, cmap='Blues', fmt=".2%")

ax_train.set_title('QCNN Training Confusion Matrix\n\n');
ax_train.set_xlabel('\nPredicted Values')
ax_train.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax_train.xaxis.set_ticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax_train.yaxis.set_ticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

cf_val_matrix = confusion_matrix(y_val, val_prediction.argmax(axis=-1))

cf_val_percent = np.zeros([len(cf_val_matrix), len(cf_val_matrix[0])])
for idx in range(len(cf_val_matrix)):
    cf_val_percent[idx] = cf_val_matrix[idx] / np.sum(cf_val_matrix[idx])
    
ax_val = sns.heatmap(cf_val_percent, annot=True, cmap='Blues', fmt=".2%")

ax_val.set_title('QCNN Validation Confusion Matrix\n\n');
ax_val.set_xlabel('\nPredicted Values')
ax_val.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax_val.xaxis.set_ticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax_val.yaxis.set_ticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

cf_test_matrix = confusion_matrix(y_test, test_prediction.argmax(axis=-1))    

cf_test_percent = np.zeros([len(cf_test_matrix), len(cf_test_matrix[0])])
for idx in range(len(cf_test_matrix)):
    cf_test_percent[idx] = cf_test_matrix[idx] / np.sum(cf_test_matrix[idx])
    
ax_test = sns.heatmap(cf_test_percent, annot=True, cmap='Blues', fmt=".2%")

ax_test.set_title('QCNN Test Confusion Matrix\n\n');
ax_test.set_xlabel('\nPredicted Values')
ax_test.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax_test.xaxis.set_ticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax_test.yaxis.set_ticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()