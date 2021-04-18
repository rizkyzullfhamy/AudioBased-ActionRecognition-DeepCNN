#Import library yang dibutuhkan dalam sistem
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import zipfile,os, shutil
from distutils.dir_util import copy_tree

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
<<<<<<< HEAD


#Inisial folder untuk dataset 
base_dir = "C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Dataset/Image/"
train_dir = os.path.join(base_dir, 'd_train')
validation_dir = os.path.join(base_dir, 'd_val')

# Membuat direktori data training 
train_askhelping_dir     = os.path.join(train_dir, 'AskHelping')
train_brokenglass_dir    = os.path.join(train_dir, 'BrokenGlass')
train_cooking_dir        = os.path.join(train_dir, 'Cooking')
train_crying_dir         = os.path.join(train_dir, 'Crying')
train_eating_dir         = os.path.join(train_dir, 'Eating')
train_laughing_dir       = os.path.join(train_dir, 'Laughing')
train_listenmusic_dir    = os.path.join(train_dir, 'ListenMusic')
train_screamofpain_dir   = os.path.join(train_dir, 'ScreamofPain')
train_washingclothes_dir = os.path.join(train_dir, 'WashingClothes')
train_washinghand_dir    = os.path.join(train_dir, 'WashingHand')
train_watchingtv_dir     = os.path.join(train_dir, 'WatchingTV')

# Membuat direktori data validasi
validation_askhelping_dir     = os.path.join(validation_dir, 'AskHelping')
validation_brokenglass_dir    = os.path.join(validation_dir, 'BrokenGlass')
validation_cooking_dir        = os.path.join(validation_dir, 'Cooking')
validation_crying_dir         = os.path.join(validation_dir, 'Crying')
validation_eating_dir         = os.path.join(validation_dir, 'Eating')
validation_laughing_dir       = os.path.join(validation_dir, 'Laughing')
validation_listenmusic_dir    = os.path.join(validation_dir, 'ListenMusic')
validation_screamofpain_dir   = os.path.join(validation_dir, 'ScreamofPain')
validation_washingclothes_dir = os.path.join(validation_dir, 'WashingClothes')
validation_washinghand_dir    = os.path.join(validation_dir, 'WashingHand')
validation_watchingtv_dir     = os.path.join(validation_dir, 'WatchingTV')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0,
    horizontal_flip=False,
    vertical_flip=False,
    shear_range=0.1,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0,
    horizontal_flip=False,
    vertical_flip=False,
    shear_range=0.1,
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(250,250),
    batch_size=32,
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(250,250),
    batch_size=32,
    class_mode='categorical')

# CREATE MODEL CNN
model = Sequential()

model.add(Conv2D(32,(5,5),activation='relu', input_shape=(250,250,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(64, (4,4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(256, (1,1), activation='relu'))
model.add(Conv2D(256, (4,4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(256, (2,2),activation='relu'))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(11))
model.add(Activation('softmax'))

#kompilasi model yang sebelumnya dibuat
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


#Display model architecture summary
model.summary()

#calculate pre-training accuracy
score = model.evaluate(validation_generator, verbose=0)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)


# Training model 
from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epoch = 30
num_batch_size = 30

checkpointer = ModelCheckpoint(filepath='C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Model/AudioBased_ModelCNN.h5', verbose= 1, save_best_only=True)
start = datetime.now()

model.fit(
    train_generator, 
    steps_per_epoch= num_batch_size,
    epochs= num_epoch,
    validation_data= validation_generator,
    callbacks= [checkpointer],
    validation_steps= num_batch_size,
    verbose=1)

duration = datetime.now() - start
print("Traning complete in time : ", duration)

#evaluate model on the training  and testing set
score = model.evaluate(train_generator, verbose=0)
print("Training Accuracy : ", score[1])

score = model.evaluate(validation_generator, verbose=0)
print("Testing Accuracy : ", score[1])


# saved model after training and testing dataset 
model.save("C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Model/Out_AudioBased_ModelCNN.h5")
=======
>>>>>>> parent of 535b0a2c... test
