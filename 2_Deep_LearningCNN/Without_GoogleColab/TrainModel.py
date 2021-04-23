import numpy as np 
import os 
import tensorflow as tf 
import time
from datetime import datetime
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import zipfile,os, shutil
from distutils.dir_util import copy_tree
from ConfigTensor import ConfTF

num_epoch = 80
num_batch_size = 32

# Folder initials for the dataset
base_dir = "C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Dataset/Image/"
train_dir = os.path.join(base_dir, 'd_train')
validation_dir = os.path.join(base_dir, 'd_val')

# Initialize the training data folder
train_cooking_dir        = os.path.join(train_dir, 'Cooking')
train_crying_dir         = os.path.join(train_dir, 'Crying')
train_eating_dir         = os.path.join(train_dir, 'Eating')
train_listenmusic_dir    = os.path.join(train_dir, 'ListenMusic')
train_washingclothes_dir = os.path.join(train_dir, 'WashingClothes')
train_washinghand_dir    = os.path.join(train_dir, 'WashingHand')
train_watchingtv_dir     = os.path.join(train_dir, 'WatchingTV')
# train_askhelping_dir     = os.path.join(train_dir, 'AskHelping')
# train_brokenglass_dir    = os.path.join(train_dir, 'BrokenGlass')
# train_laughing_dir       = os.path.join(train_dir, 'Laughing')
# train_screamofpain_dir   = os.path.join(train_dir, 'ScreamofPain')

# Initialize the validation data folder
validation_cooking_dir        = os.path.join(validation_dir, 'Cooking')
validation_crying_dir         = os.path.join(validation_dir, 'Crying')
validation_eating_dir         = os.path.join(validation_dir, 'Eating')
validation_listenmusic_dir    = os.path.join(validation_dir, 'ListenMusic')
validation_washingclothes_dir = os.path.join(validation_dir, 'WashingClothes')
validation_washinghand_dir    = os.path.join(validation_dir, 'WashingHand')
validation_watchingtv_dir     = os.path.join(validation_dir, 'WatchingTV')
# validation_askhelping_dir     = os.path.join(validation_dir, 'AskHelping')
# validation_brokenglass_dir    = os.path.join(validation_dir, 'BrokenGlass')
# validation_laughing_dir       = os.path.join(validation_dir, 'Laughing')
# validation_screamofpain_dir   = os.path.join(validation_dir, 'ScreamofPain')

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
def training(num_batch_size, num_epoch, loadmodel_dir,modelsaved):
    print("\nLoad Model CNN\n")
    model = tf.keras.models.load_model(loadmodel_dir)
    time.sleep(1)

    checkpointer = ModelCheckpoint(filepath='C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Model/AudioBased_ModelCNN.h5', verbose= 1, save_best_only=True)
    start = datetime.now()

    print("\nStart Training data ...\n")
    history = model.fit(
        train_generator,
        steps_per_epoch=num_batch_size,
        epochs=num_epoch,
        validation_data= validation_generator,
        callbacks = [checkpointer],
        validation_steps = num_batch_size,
        verbose = 1 )

    duration = datetime.now() - start
    print("\nTraning complete in time : ", duration)

    #evaluate model on the training  and testing set
    score = model.evaluate(train_generator, verbose=0)
    print("\nTraining Accuracy : ", score[1])

    score = model.evaluate(validation_generator, verbose=0)
    print("\nTesting Accuracy : ", score[1])

    # saved model after training and testing dataset 
    print("\nThe trained CNN models are complete\n")
    model.save(modelsaved)
    # time.sleep(3)
    return history

dirmodel = 'C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Model/ModelCNN1.h5'
dirsaved = "C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Model/ModelCNN_AfterTrain2_1.h5"
#Load Config Tensorflow
ConfTF()
#Training
history = training(num_batch_size,num_epoch,dirmodel,dirsaved)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

