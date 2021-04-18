# Import libraries as needed
import argparse
import tempfile
import queue
import sys
import math
import os.path
import time

import sounddevice as sd 
import soundfile as sf 

import numpy as np 
import librosa
import librosa.display
from matplotlib import pyplot as plt 

saved_audio = '/home/pi/ProyekAkhir/Dataset/audioListenMusic'
saved_featureImage = '/home/pi/ProyekAkhir/Dataset/imageWatchingTV'
durasi_Rec = 5
channels_Sen = 4
device_Sen = 2 #ID device sensor
fs = 8000
count = 619

def AudioRec(duration, fs, channel, deviceId, saveDir,filename,count):
    sd.default.samplerate = fs
    sd.default.channels   = channel
    sd.default.device     = deviceId
    recordAudio           = sd.rec(int(duration*fs))
    print("Record...\n")
    sd.wait()
    print("Done record.....\n")
    name_file = ('audio'+filename+'_'+str(count)+'.wav')
    completeName = os.path.join(saveDir, name_file)
    sf.write(completeName, recordAudio, fs)
    print("Saved")
    return completeName

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ExtractFeature(path_audio, dir_saved,filename,count):
    data, sr = librosa.load(path_audio+'/audio'+filename+'_'+str(count)+'.wav', sr=None, mono=True, offset=0.0, duration=None)
    print(len(data), sr)
    #extracting mel spectrogram feature
    mel_spectrogram = librosa.feature.melspectrogram(data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    print("MEL SPECTROGRAM : ",mel_spectrogram.shape)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    print("LOG MEL SPECTROGRAM : ",log_mel_spectrogram.shape)
    Norm_MelSpectrogram = NormalizeData(log_mel_spectrogram)
    librosa.display.specshow(Norm_MelSpectrogram, sr=sr, x_axis="time", y_axis="mel")
    # plt.colorbar()
    # plt.show()
    #plt.figure(str(count))
    saved_file = (dir_saved + '/plot'+ filename +str(count)+'.png')
    plt.savefig(saved_file)
    # return saved_file

if __name__ == "__main__":
    # filename = input("name file audio : ")
    filename = 'ListenMusic'
    while(True):
        print(count)
        audioPath = AudioRec(durasi_Rec, fs, channels_Sen, device_Sen, saved_audio,filename, count)
        data,sr = librosa.load(audioPath, sr=None, mono=True, offset=0.0, duration=None)
        rms_result = math.sqrt(np.mean(data*data))
        print("RMS RESULT : ",rms_result*1000)
        # ExtractFeature(saved_audio, saved_featureImage, filename,count)
        count += 1
        if(count > 1500):
            break
