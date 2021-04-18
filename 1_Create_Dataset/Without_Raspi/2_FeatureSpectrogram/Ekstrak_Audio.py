import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os 

count = 0
batas = 0
tick = 0
def normalize_spectrogram( Y, sr, hop_length, eps=1e-6):
    mean = Y.mean()
    std = Y.std()
    spec_norm = (Y - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    np.seterr(divide='ignore', invalid='ignore')
    spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
    #spec_scaled = spec_scaled.astype(np.uint8)
    # normalize the waveform
    np.seterr(divide='ignore', invalid='ignore')
    norm_sample = (spec_scaled - spec_scaled.mean()) / spec_scaled.std()
    return norm_sample
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_spectrogram(path_audio, count, dir_saved,filename):
    data,sr = librosa.load(path_audio, sr=None,mono=True, offset=0.0, duration=None)
    print(len(data), sr)
    #Mel Filter Banks
    #fil_banks = librosa.filters.mel(n_fft=2048, sr=sr, n_mels=128)
    #print(fil_banks.shape)
    # plt.figure(1)
    # librosa.display.specshow(fil_banks,sr=sr,x_axis="linear")
    # plt.colorbar(format="%+2.f")
    #Extracting Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(data, sr=sr, n_fft=2048, hop_length=512,n_mels=128)
    print(mel_spectrogram.shape)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    print(log_mel_spectrogram.shape)
    Norm_MelSpectrogram = NormalizeData(log_mel_spectrogram)
    #Norm_MelSpectrogram = normalize_spectrogram(log_mel_spectrogram, sr=sr, hop_length=512)
    checkNan = np.sum(Norm_MelSpectrogram)
    if(np.isnan(checkNan)== True):
        Norm_MelSpectrogram = np.zeros((128,79))
        print("TRUE")
    plt.figure(str(count))
    print("NormSpec demension : ", Norm_MelSpectrogram.shape)
    test = np.zeros((128,79))
    tick = 0
    tick1 = 0
    while True:
        if (tick1 >= len(Norm_MelSpectrogram[0])):        
            tick1 = 0
            tick += 1
            if(tick >= len(Norm_MelSpectrogram)):
                break
        #print(tick,tick1)
        test[tick][tick1] = Norm_MelSpectrogram[tick][tick1]
        tick1 += 1
    print("Test : ", test.shape)
    print(test)
    librosa.display.specshow(test, sr=sr, x_axis="time", y_axis="mel")
    # plt.colorbar()
    plt.savefig(dir_saved +'/'+ 'plot'+ (filename) + str(count) + '.png')
    #plt.close()
    # plt.show()

path_audio = input('Directory file .wav yang akan diekstra : ')
path_savespec = input('Directory save spectrogram : ')
filename = input('Filename audio or plot : ')
# Melihat isi file dalam directory
path, dirs, files = next(os.walk(path_audio))
file_count = len(files) 
batas = file_count
# save_spectrogram(path_audio + "/audio0-05-1.wav", count, path_savespec) =>Test spectrogram tunggal
while count < batas:
    print(count)
    save_spectrogram(path_audio + "/audio"+ (filename) + "_" + str(count) + ".wav", count, path_savespec,filename)
    count += 1
    
# C:/Users/Rizky Zull Fhamy/Documents/Bismillah_TA/Dataset/SplitAudio_5s/Audiosplit_Result/audioWatchingTV_0
# C:/Users/Rizky Zull Fhamy/Documents/Bismillah_TA/Dataset/EkstrakFitur_Audio/Spectrogram_Result/Spectrogram_WatchingTV_0