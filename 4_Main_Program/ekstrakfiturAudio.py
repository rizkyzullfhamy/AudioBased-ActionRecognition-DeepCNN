import numpy as np 
import librosa
import librosa.display
from matplotlib import pyplot as plt 
import time

audioPath = '/home/pi/ProyekAkhir/ResultAudio'
saved_featureImage = '/home/pi/ProyekAkhir/ResultImageFeature'
count = 0

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ExtractFeature(path_audio, dir_saved,count):
    data, sr = librosa.load(path_audio+'/audioNoActivity_'+str(count)+'.wav', sr=None, mono=True, offset=0.0, duration=None)
    print(len(data), sr)
    plt.plot(data)
    # #extracting mel spectrogram feature
    # mel_spectrogram = librosa.feature.melspectrogram(data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    # print("MEL SPECTROGRAM : ",mel_spectrogram.shape)
    # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    # print("LOG MEL SPECTROGRAM : ",log_mel_spectrogram.shape)
    # Norm_MelSpectrogram = NormalizeData(log_mel_spectrogram)
    # librosa.display.specshow(Norm_MelSpectrogram, sr=sr, x_axis="time", y_axis="mel")
    #plt.colorbar()
    plt.show()
    time.sleep(3)
    plt.close()
    #plt.figure(str(count))
    # saved_file = (dir_saved + '/imagefeature_'+str(count)+'.png')
    # plt.savefig(saved_file)
    # return saved_file

if __name__ == "__main__":

    while(True):
        print(count,'\n')
        imageFeaturePath = ExtractFeature(audioPath, saved_featureImage, count)
        count += 1