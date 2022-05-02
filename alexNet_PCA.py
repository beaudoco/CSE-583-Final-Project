import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout
# from tensorflow.keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler

# from keras.utils.np_utils import to_categorical

# import pandas as pd
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder 
# from sklearn.preprocessing import StandardScaler
# import keras
import numpy as np
# from scipy.linalg import eigh

from skimage.transform import resize
import IPython.display as display
import PIL.Image as im

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

PREPROCESS = True
SAVE_PATH = "pca/" # Data saving folder

if PREPROCESS == True:
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
    
    x_train = resizeArr(X_train_pca)
    x_test = resizeArr(X_test_pca)

    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)

    x_train = tf.cast(x_train, dtype=tf.float16)
    x_test = tf.cast(x_test, dtype=tf.float16)

    x_train = tf.image.grayscale_to_rgb(x_train)
    x_test = tf.image.grayscale_to_rgb(x_test)

    pca_train_images = np.asarray(x_train)
    pca_test_images = np.asarray(x_test)

    np.save(SAVE_PATH + "pca_train_images.npy", pca_train_images)
    np.save(SAVE_PATH + "pca_test_images.npy", pca_test_images)

x_train = np.load(SAVE_PATH + "pca_train_images.npy")
x_test = np.load(SAVE_PATH + "pca_test_images.npy")

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

def download(max_dim=None):
#   pca = PCA(196)
#   print(X_train_pca.shape)
#   quit()
#   recovered = pca.inverse_transform(X_train_pca)
#   img = recovered[1,:].reshape([28,28])
#   plt.imshow(img, cmap='gray_r')
#   plt.show()
  img = x_val[-1:]
  img = (img * 255).astype(np.uint8)
  np.reshape(img, (224, 224, 3))
  img = np.reshape(img, (224, 224, 3))
  img = im.fromarray(img)
  # plt.imshow(img)
  # plt.show()
#   print(img)
  # quit()
#   img = img.convert("RGB")
  if max_dim:
    img.thumbnail((max_dim, max_dim))
  return np.array(img)

# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

# Downsizing the image makes it easier to work with.
original_img = download(max_dim=224)

names = ['conv2d_3', 'conv2d_4']
layers = [model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=model.input, outputs=layers)

def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)

class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, img, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = calc_loss(img, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

      return loss, img

deepdream = DeepDream(dream_model)

def run_deep_dream(img, steps=100, step_size=0.01):
  # Convert from uint8 to the range expected by the model.
#   print(img)
  img = tf.keras.applications.inception_v3.preprocess_input(img)
#   print(img)
#   quit()
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    loss, img = deepdream(img, run_steps, tf.constant(step_size))

    display.clear_output(wait=True)
    print ("Step {}, loss {}".format(step, loss))


  result = deprocess(img)
  display.clear_output(wait=True)
  plt.imshow(result)
  plt.show()
#   img = result.reshape([28,28])
#   print(result)
  result = tf.image.rgb_to_grayscale(result)
  img = result.numpy()
  tmp_img = resize(img,(1, 196))
  hold = tmp_img.reshape((1, 196))
  recovered = pca.inverse_transform(hold)
  img = recovered.reshape([28,28])
  plt.imshow(img, cmap='gray_r')
  plt.show()

  return result

dream_img = run_deep_dream(img=original_img, steps=100, step_size=0.01)