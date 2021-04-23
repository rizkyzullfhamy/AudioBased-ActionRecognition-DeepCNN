# Program integrasi sistem secara keseluruhan secara realtime#

# Import libraries as needed
import argparse
import tempfile
import queue
import sys
import os.path
import time

import sounddevice as sd 
import soundfile as sf 
from pydub import AudioSegment

import numpy as np 
import librosa
import librosa.display
from matplotlib import pyplot as plt 

import tensorflow as tf 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import Rms
import pubmqtt


######################## Function def program ####################
def AudioRec(duration, fs, channel, deviceId, saveDir):
    sd.default.samplerate = fs
    sd.default.channels   = channel
    sd.default.device     = deviceId
    recordAudio           = sd.rec(int(duration*fs))
    print("Record...")
    sd.wait()
    print("Done record...")
    name_file = ('DataAudio.wav')
    completeName = os.path.join(saveDir, name_file)
    sf.write(completeName, recordAudio, fs)
    print("Saved")
    return completeName

def splitaudio(path_dir, dir_audio):
    audio = AudioSegment.from_file(dir_audio)
    audio = audio.set_channels(1)
    lengthaudio = len(audio)
    print("Length of Audio File " ,lengthaudio)
    start = 1000
    end = 0
    while start < len(audio):
        end += threshold
        print(start,end)
        chunk = audio[start:end]
        filename = f'{path_dir}/DataAudio1.wav'
        chunk.export(filename, format="wav")
        start += threshold
    return filename

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ExtractFeature(path_audio, dir_saved):
    data, sr = librosa.load(path_audio, sr=None, mono=True, offset=0.0, duration=None)
    print(len(data), sr)
    # RMS energy audio signal => untuk mengecek ada aktivitas suara atau tidak
    Rms_Result = Rms.Rms_Audio(data, sr)
    if (Rms_Result < threshld):
        print("\nNo Activity\n")
        flagg = 0
    else:
        #extracting mel spectrogram feature
        mel_spectrogram = librosa.feature.melspectrogram(data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        print("MEL SPECTROGRAM : ",mel_spectrogram.shape)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        print("LOG MEL SPECTROGRAM : ",log_mel_spectrogram.shape)
        Norm_MelSpectrogram = NormalizeData(log_mel_spectrogram)
        librosa.display.specshow(Norm_MelSpectrogram, sr=sr, x_axis="time", y_axis="mel")
        #plt.colorbar()
        #plt.show()
        saved_file = (dir_saved + '/imagefeature.png')
        plt.savefig(saved_file)
        flagg = 1
    return saved_file,flagg


def printPrediction(file_InputImage):
    InImage = image.load_img(file_InputImage, target_size=(250,250))
    x = image.img_to_array(InImage)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    print(x.shape)

    predicted_vector = np.argmax(model.predict(x), axis=1)
    predicted_class  = label[(predicted_vector[0])]
    print("The Predicted Class is : ", predicted_class,'\n')

    predicted_proba_vector = model.predict(x)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        print(label[i], ": ", format(predicted_proba[i], '.12f'))
        print("\n\n")
    
    return predicted_class

################## Init program ########################
#class label
label = ["Cooking", "Crying", "Eating", "ListenMusic", "WashingClothes", "WashingHand", "WatchingTV"]

saved_audio = '/home/pi/ProyekAkhir/ResultAudio'
saved_featureImage = '/home/pi/ProyekAkhir/ResultImageFeature'
dir_SavedSpectogram = "C:/Users/LENOVO/Documents/RizkyZullFhamy/AudioBased-ActionRecognition-DeepCNN/3_Audio_Image/Result_Image/InputSpectogram.png"
durasi_Rec = 6
channels_Sen = 4
device_Sen = 2 #ID device sensor
fs = 8000
count = 0
Rms_Result = 0.0
flagg = 0
threshld = 7
threshold = 6000     # In Milliseconds, this will cut 6 Sec of audio

# Load Model
print("Load Model...")
model = tf.keras.models.load_model('/home/pi/ProyekAkhir/Model/mymodel_new.h5')
print("Load Model Succes")
time.sleep(2)

################# Loop program ########################
if __name__ == "__main__":
    print("Start System ...")
    while(True):
        audioPath = AudioRec(durasi_Rec, fs, channels_Sen, device_Sen, saved_audio)
        audioNewPath = splitaudio(saved_audio,audioPath) 
        imageFeaturePath, flagg = ExtractFeature(audioNewPath, saved_featureImage)
        if (flagg != 0 ):
            Label_Predict = printPrediction(imageFeaturePath)
            pubmqtt.publish_MQTT('103.106.72.187', '/action/aurecog/5f8f2dd0c00456e8e03e5e9c', dir_SavedSpectogram, Label_Predict)

