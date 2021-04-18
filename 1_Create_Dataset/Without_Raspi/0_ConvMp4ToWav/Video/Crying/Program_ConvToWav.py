import os
import ffmpy
import ffmpeg 
 
# File dir Video
inputdir = input('Directory Video yang akan di konversi : ')
outdir = input('Directory Audio yang telah dikonversi : ')
# inputdir = 'C:/Users/Rizky Zull Fhamy/Documents/Bismillah_TA/Dataset/ConvMp4ToWav/Video/WatchingTV'
# outdir   = '../../Audio/WatchingTV'
for filename in os.listdir(inputdir):
    actual_filename = filename[:-4]
    if(filename.endswith(".mp3") or (".wav") or (".flac") or (".mp4")):
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 8000 {}/{}.wav'.format(filename,outdir, actual_filename))
    else:
        continue