import argparse
import tempfile
import queue
import sys
import os.path
import time

import sounddevice as sd 
import soundfile as sf 

saved_audio = '/home/pi/ProyekAkhir/audioNoActivity'
durasi_Rec = 5
channels_Sen = 4
device_Sen = 2
fs = 8000
count = 0

def AudioRec(duration, fs, channel, deviceId, saveDir,count):
    sd.default.samplerate = fs
    sd.default.channels   = channel
    sd.default.device     = deviceId
    recordAudio           = sd.rec(int(duration*fs))
    print("Record...\n")
    sd.wait()
    print("Done record.....\n")
    name_file = ('audioNoActivity_'+str(count)+'.wav')
    completeName = os.path.join(saveDir, name_file)
    sf.write(completeName, recordAudio, fs)
    print("Saved")
    return completeName

if __name__ == "__main__":

    while(True):
        print(count)
        audioPath = AudioRec(durasi_Rec, fs, channels_Sen, device_Sen, saved_audio,count)
        count += 1