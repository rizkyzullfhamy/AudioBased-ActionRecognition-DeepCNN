import numpy as np 
import os 
import tensorflow as tf 
import time
from ConfigTensor import ConfTF
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import zipfile,os, shutil
from distutils.dir_util import copy_tree

#Load Config Tensorflow
ConfTF()

base_dir = "C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Dataset/Image/"
validation_dir = os.path.join(base_dir, 'd_val')

# Membuat direktori data validasi
# validation_askhelping_dir     = os.path.join(validation_dir, 'AskHelping')
# validation_brokenglass_dir    = os.path.join(validation_dir, 'BrokenGlass')
validation_cooking_dir        = os.path.join(validation_dir, 'Cooking')
validation_crying_dir         = os.path.join(validation_dir, 'Crying')
validation_eating_dir         = os.path.join(validation_dir, 'Eating')
# validation_laughing_dir       = os.path.join(validation_dir, 'Laughing')
validation_listenmusic_dir    = os.path.join(validation_dir, 'ListenMusic')
# validation_screamofpain_dir   = os.path.join(validation_dir, 'ScreamofPain')
validation_washingclothes_dir = os.path.join(validation_dir, 'WashingClothes')
validation_washinghand_dir    = os.path.join(validation_dir, 'WashingHand')
validation_watchingtv_dir     = os.path.join(validation_dir, 'WatchingTV')

test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0,
    horizontal_flip=False,
    vertical_flip=False,
    shear_range=0.1,
    fill_mode='nearest')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(250,250),
    batch_size=32,
    class_mode='categorical')


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

model.add(Dense(7))                 # Ubah Sesuai Rekog yang diinginkan
model.add(Activation('softmax'))

print("\nCreating CNN models successfully\n")
time.sleep(3)

print("\nCompile CNN models\n")
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
time.sleep(3)

print("\nDisplay model architecture summary\n")
model.summary()

#calculate pre-training accuracy
score = model.evaluate(validation_generator, verbose=0)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)    

# saved model after training and testing dataset 
model.save("C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Model/ModelCNN.h5")