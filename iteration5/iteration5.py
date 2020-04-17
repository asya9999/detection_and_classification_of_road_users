import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import math

from sklearn.decomposition import PCA
from PIL import Image, ImageOps

from skimage import data
from skimage.color import rgb2gray

"""Import data from drive"""

from google.colab import drive
drive.mount('/content/drive')

import zipfile

zip_ref = zipfile.ZipFile("/content/drive/My Drive/stanford-car-dataset-by-classes-folder.zip", 'r')
zip_ref.extractall()
zip_ref.close()

"""Read data and take paths"""

path_to_train = "/content/stanford-car-dataset-by-classes-folder/car_data/car_data/train"
path_to_test = "/content/stanford-car-dataset-by-classes-folder/car_data/car_data/test"

path_to_names = "/content/stanford-car-dataset-by-classes-folder/names.csv"
path_to_train_an = "/content/stanford-car-dataset-by-classes-folder/anno_train.csv"
path_to_test_an = "/content/stanford-car-dataset-by-classes-folder/anno_test.csv"

names = pd.read_csv(path_to_names)
names.head()

train_an = pd.read_csv(path_to_train_an)
train_an.rename(columns = {
    '00001.jpg':'name',
    '39':'hb',
    '116':'vb',
    '569':'ht',
    '375':'vt',
    '14':'car_type'
    }, inplace = True) 
#train_an.set_index("name", inplace = True)
train_an.head()

test_an = pd.read_csv(path_to_test_an)
test_an.rename(columns = {
    '00001.jpg':'name',
    '30':'hb',
    '52':'vb',
    '246':'ht',
    '147':'vt',
    '181':'car_type'
    }, inplace = True) 
#test_an.set_index("name", inplace = True)
test_an.head()

print(train_an.iloc[2361])

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

import cv2 as cv
#image normalization
def normalize_image(im_array):
    normalizedImg = np.zeros(im_array.shape)
    #normalizedImg = cv.normalize(array, normalizedImg, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    normalizedImg = cv.normalize(im_array,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    return normalizedImg

def make_padding(image_chosen):
    h, w, c = image_chosen.shape
    if w>h:
      to_append = np.zeros(((abs(w-h)//2), w, 3))
      new_image =  np.append(to_append, image_chosen,axis = 0)
      to_append = np.zeros((abs(w-h) - abs(w-h)//2, w, 3))
      new_image = np.append(new_image, to_append,axis = 0)
      image_chosen = new_image
    if h>w:
      to_append = np.zeros((h, (abs(w-h)//2), 3))
      new_image =  np.append(to_append, image_chosen,axis = 1)
      to_append = np.zeros((h, abs(w-h) - (abs(w-h)//2), 3))
      new_image = np.append(new_image, to_append,axis = 1)
      image = new_image

    #print(image_chosen)
    return image_chosen

def prepare_data(path_to_images, ann):
  paths = []
  images = []
  classes = []
  for root, dirs, files in os.walk(path_to_images):
    for f in files:
      #find index
      index = int(f[:-4]) - 2
      img=mpimg.imread(root + "/" + f)
      paths.append(root + "/" + f)

      #check if image is grey
      if(len(img.shape)<3):
        img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
      
      #plt.imshow(img)
      #plt.show()

      #take image from bounding boxes
      annotation = ann.iloc[index]
      img = img[annotation['vb']:annotation['vt'], annotation['hb']:annotation['ht'], :]
      #plt.imshow(img)
      #plt.show()

      #make crop
      img = make_padding(img)
      #plt.imshow(img)
      #plt.show()


      #resize
      chosen_size = (50, 50)
      img = Image.fromarray(img.astype('uint8'), 'RGB')
      img = ImageOps.fit(img, chosen_size, Image.ANTIALIAS)
      #plt.imshow(img)
      #plt.show()

      #normalize
      img = normalize_image(np.array(img))
      #plt.imshow(img)
      #plt.show()

      #Add to array
      images.append(img)
      classes.append(annotation['car_type'])

      break
 
  images = np.array(images)
  return images, classes

trainX, trainY = prepare_data(path_to_train, train_an)

testX, testY = prepare_data(path_to_test, test_an)

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

print(trainX.shape)
print(trainY.shape)

#make from image vector
def make_vector(array):
    images = []
    for i in array:
        images.append(i.flatten())
    return np.array(images)

trainX = np.array((trainX))
testX = np.array((testX))

#trainY = np.array(trainY)
#testY = np.array(testY)
trainX = trainX.reshape((trainX.shape[0], 10000))
testX = testX.reshape((testX.shape[0], 10000))

from keras.utils import to_categorical
trainY = to_categorical(trainY)
testY = to_categorical(testY)

print(trainX[0, 5500])
print(testX[0, 5500])

trainY.shape

testY.shape

"""Try SVM"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

svm_classifier = SVC(decision_function_shape='ovo',verbose=True)
svm_classifier.fit(trainX, trainY)

y_pred=svm_classifier.predict(testX)
print("Accuracy:", metrics.accuracy_score(testY, y_pred))

print()

"""Neural network classifier"""

!pip install -q pyyaml h5py

import tensorflow as tf
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras


resut_accuracy__NN_train = []
resut_accuracy__NN_test = []

def train_NN(trainX, trainY, testX, testY):
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    model_name = "NN_classifier"

    checkpoint_path = F"/content/drive/My Drive/{model_name}" 
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Создаем коллбек сохраняющий веса модели
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period=100)


    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        resut_accuracy__NN_train.append(logs.get('acc'))

        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        resut_accuracy__NN_test.append(acc)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        #if(logs.get('acc')>0.94):
        #print("\nReached 94% accuracy!")
        #self.model.stop_training = True
        
    (x_train, y_train),(x_test, y_test) = (np.array(trainX), np.array(trainY)), (np.array(testX), np.array(testY))

    training_images = x_train
    test_images = x_test
    
    callback = myCallback()
    dropout = 0.25 # Dropout, probability to drop a unit

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(10000,)), 
        tf.keras.layers.Dense(7000, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.25),
        #fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        tf.keras.layers.Dense(3000, activation=tf.nn.relu),
        #tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu), 
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(800, activation=tf.nn.relu),
        #tf.keras.layers.Dropout(0.25),
        #tf.keras.layers.Dense(500, activation=tf.nn.relu), 
        tf.keras.layers.Dense(500, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.25), 
        tf.keras.layers.Dense(300, activation=tf.nn.relu),
        #tf.keras.layers.Dropout(0.25), 
        tf.keras.layers.Dense(197, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    model.summary()
    
    # model fitting
    history = model.fit(
        
        training_images, y_train, epochs=500, callbacks = [callback, cp_callback]
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]

train_NN(trainX, trainY, testX, testY)

"""CNN classifier"""

#train_it = datagen.flow(trainX, trainY)
#test_it = datagen.flow(testX, testY)
from keras.optimizers import RMSprop
import keras
from keras.layers.core import Layer
import keras.backend as K

BS = 32

aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
  brightness_range=(0.1, 0.9),
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
total = 0

resut_accuracy__CNN_train = []
resut_accuracy__CNN_test = []

trainX = trainX.reshape((trainX.shape[0], 50, 50, 3))
testX = testX.reshape((testX.shape[0], 50, 50, 3))

print(trainX[0][50][50])

def train_CNN(trainX, trainY, testX, testY):
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    model_name = "CNN_classifier_n"

    checkpoint_path = F"/content/drive/My Drive/{model_name}" 
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Создаем коллбек сохраняющий веса модели
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period=100)


    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        resut_accuracy__CNN_train.append(logs.get('acc'))

        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        resut_accuracy__CNN_test.append(acc)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        #if(logs.get('acc')>0.94):
        #print("\nReached 94% accuracy!")
        #self.model.stop_training = True
        
    (x_train, y_train),(x_test, y_test) = (np.array(trainX), np.array(trainY)), (np.array(testX), np.array(testY))

    training_images = x_train
    test_images = x_test
    
    callback = myCallback()
    dropout = 0.25 # Dropout, probability to drop a unit

    model = tf.keras.models.Sequential([
                                        

        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', input_shape=(50, 50, 3)),

        tf.keras.layers.Conv2D(512, (1, 1), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(512, (1, 1),  padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, (1, 1), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(512, (1, 1), padding='same',  activation='relu'),

        tf.keras.layers.MaxPooling2D(2, 2),


        tf.keras.layers.Conv2D(128, (3, 3), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same',  activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(256, (3, 3), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3),  padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same',  activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),


        tf.keras.layers.Conv2D(512, (3, 3), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3),  padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same',  activation='relu'),

        tf.keras.layers.MaxPooling2D(2, 2),


        tf.keras.layers.Conv2D(512, (2, 2), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(512, (2, 2), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(512, (2, 2), padding='same',  activation='relu'),
        tf.keras.layers.Conv2D(512, (2, 2), padding='same',  activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),


        tf.keras.layers.Flatten(), 

        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(500, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.25), 
     
     
        tf.keras.layers.Dense(197, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    model.summary()
    
    # model fitting
    history = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        #trainX, 
        #[trainY, trainY, trainY],
        steps_per_epoch=len(trainX)//BS,
        epochs=2000,
        callbacks = [callback, cp_callback]
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]

train_CNN(trainX, trainY, testX, testY)

#train_it = datagen.flow(trainX, trainY)
#test_it = datagen.flow(testX, testY)
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D,  Flatten

import numpy as np 
from keras.datasets import cifar10 
from keras import backend as K 
from keras.utils import np_utils

import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint

BS = 12

aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
  brightness_range=(0.1, 0.9),
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
total = 0

resut_accuracy__insCNN_train = []
resut_accuracy__insCNN_test = []

trainX = trainX.reshape((trainX.shape[0], 50, 50, 3))
testX = testX.reshape((testX.shape[0], 50, 50, 3))


kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3,  pool_proj], axis=3, name=name)
    
    return output

# custom generator
def multiple_outputs(generator, trainX, y_train, BS):
    gen = generator.flow(trainX, y_train, batch_size=BS)
    while True:
        gnext = gen.next()
        # return image batch and 3 sets of lables
        yield gnext[0], [gnext[1], gnext[1], gnext[1]]

def train_insCNN(trainX, trainY, testX, testY):
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    model_name = "insCNN_classifier"

    checkpoint_path = F"/content/drive/My Drive/{model_name}" 
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Создаем коллбек сохраняющий веса модели
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period=100)


    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        resut_accuracy__insCNN_train.append(logs.get('acc'))

        loss =  self.model.evaluate(x_test, [y_test, y_test, y_test])
        resut_accuracy__insCNN_test.append(loss)
        print(loss)
        #if(logs.get('acc')>0.94):
        #print("\nReached 94% accuracy!")
        #self.model.stop_training = True
        
    (x_train, y_train),(x_test, y_test) = (np.array(trainX), np.array(trainY)), (np.array(testX), np.array(testY))

    print(y_train.shape)
    input_layer = trainX
    training_images = x_train
    test_images = x_test
   
    
    callback = myCallback()
    dropout = 0.25 # Dropout, probability to drop a unit

    input_layer = Input(shape=(50, 50, 3))

    x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((2, 2), padding='same',strides=(1, 1), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((2, 2), padding='same', strides=(1, 1), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=64,
                        filters_3x3_reduce=96,
                        filters_3x3=128,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        name='inception_3a')
    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=192,
                        filters_5x5_reduce=32,
                        filters_5x5=96,
                        filters_pool_proj=64,
                        name='inception_3b')

    #x = MaxPool2D((2, 2), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    x1 = AveragePooling2D((2, 2), strides=1)(x)
    x1 = Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(197, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4b')
    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=256,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4c')


    x2 = AveragePooling2D((2, 2), strides=3)(x)
    x2 = Conv2D(128, (3, 3), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(197, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_4e')
    x = inception_module(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4z')

    x = MaxPool2D((2, 2), padding='same', strides=(1, 1), name='max_pool_4_3x3/2')(x)

    x = Dropout(0.4)(x)

    x = Dense(197, activation='softmax', name='output')(x)

    model = Model(input_layer, [x, x1, x2], name='inception_v1')

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    model.summary()
    
    # model fitting
    history = model.fit_generator(
        #trainX, 
        #[trainY, trainY, trainY],
        #aug.flow(trainX, y_train, batch_size=BS),
        multiple_outputs(aug, trainX, y_train, BS),
        #batch_size=BS,
        steps_per_epoch=len(trainX)//BS,
        epochs=2000,
        callbacks = [callback, cp_callback]
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]

train_insCNN(trainX, trainY, testX, testY)
