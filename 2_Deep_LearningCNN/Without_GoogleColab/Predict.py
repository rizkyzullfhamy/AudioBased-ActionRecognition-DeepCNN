import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import zipfile,os, shutil
from distutils.dir_util import copy_tree
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

dir_audio = "C:/Users/LENOVO/Documents/RizkyZullFhamy/Bismillah_TA/Dataset/SplitAudio_5s/Audiosplit_Result/audioListenMusic"
dir_saved = "C:/Users/LENOVO/Documents/RizkyZullFhamy/AudioBased-ActionRecognition-DeepCNN/3_Audio_Image/Result_Image"
label = ["Cooking", "Crying", "Eating", "ListenMusic", "WashingClothes", "WashingHand", "WatchingTV"]
model = tf.keras.models.load_model("C:/Users/LENOVO/Documents/RizkyZullFhamy/TA/Model/ModelCNN_AfterTrain2.h5")


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ExtractFeatures(filename):
    data,sr = librosa.load(filename, sr=None, mono=True, offset=0.0, duration=None)
    print(len(data), sr)
    # energy = np.max(data.astype(float)**2)
    # print("ENERGI SIGNAL : ", energy)
    mel_spectrogram = librosa.feature.melspectrogram(data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    print(mel_spectrogram.shape)
    log_melspectrogram = librosa.power_to_db(mel_spectrogram)
    # print("energymin : ", np.min(log_melspectrogram))
    # print("energymax : ", np.max(log_melspectrogram))
    # print("energiRatarata : ", np.average(log_melspectrogram))
    print(log_melspectrogram.shape)
    Norm_melspectrogram = NormalizeData(log_melspectrogram)
    print(Norm_melspectrogram.shape)
    # print("energymin : ", np.min(Norm_melspectrogram))
    # print("energymax : ", np.max(Norm_melspectrogram))
    # print("energiRatarata : ", np.average(Norm_melspectrogram))
    test = np.zeros((128,79))
    tick = 0
    tick1 = 0
    while True:
        if (tick1 >= len(Norm_melspectrogram[0])):        
            tick1 = 0
            tick += 1
            if(tick >= len(Norm_melspectrogram)):
                break
            #print(tick,tick1)
        test[tick][tick1] = Norm_melspectrogram[tick][tick1]
        tick1 += 1
    print("Test : ", test.shape)
    librosa.display.specshow(test, sr=sr, x_axis="time", y_axis="mel")
    #plt.figure(1)
    plt.savefig(dir_saved+'/InputPlot.png')
    # plt.show()

def PrintPredict(filename):
    ExtractFeatures(filename)
    images = image.load_img(dir_saved+"/InputPlot.png", target_size=(250,250))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    print(x.shape)

    predicted_vector = np.argmax(model.predict(x), axis=1)
    print("\nPredic vec " , predicted_vector[0])
    predicted_class = label[(predicted_vector[0])]

    # Untuk menghitung Banyak prediksi klasifikasi
    # if(predicted_vector[0] == 6):
    #     plus = 1
    # else:
    #     plus = 0
    
    print("The predicted class is : ", predicted_class, '\n')

    predicted_proba_vector = model.predict(x)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        print(label[i], " : ", format(predicted_proba[i], '.12f'))
        #print(label[i], " : ", format(predicted_proba[i]*100, '.f'))
    
    return predicted_class

counterr = 101
LabelPredict = PrintPredict(dir_audio+"/audioListenMusic_"+str(counterr)+".wav")
