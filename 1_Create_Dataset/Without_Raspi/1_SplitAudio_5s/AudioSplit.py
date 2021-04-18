from pydub import AudioSegment
import os
 

threshold = 5000     # In Milliseconds, this will cut 5 Sec of audio
count=0           
increm = 0
#path_dir = "C:/Users/Rizky Zull Fhamy/Documents/Bismillah_TA/Dataset/SplitAudio_5s/Audiosplit_Result/"
#dir_audio = "C:/Users/Rizky Zull Fhamy/Documents/Bismillah_TA/Dataset/ConvMp4ToWav/Audio/ListenMusic/"
#filename1="audioListenMusic_"        #Ubah nama file
#filename="audioListenMusic_"

def create_dir(path_dir,filename,count):
    if os.path.isdir("Audiosplit_Result"):
        path = os.path.join(path_dir, filename + str(count)) 
        os.mkdir(path)
def split_audio(path_dir, dir_audio, filename1, count, Ccount):
    audio = AudioSegment.from_file(dir_audio + str(count) + ".wav")
    audio = audio.set_channels(1)
    lengthaudio = len(audio)
    print("Length of Audio File " + str(count) + " ",lengthaudio)
    start = 0
    end = 0
    if (count == 0):
        counter = 0
    else:
        counter = Ccount  
    while start < len(audio):
        end += threshold
        print(start,end)
        chunk = audio[start:end]
        #filename = f'{path_dir}{filename1}{count}/audio-00-{counter}.wav'
        #filename = f'{path_dir}/audio{count}-05-{counter}.wav'
        filename = f'{path_dir}/{filename1}{counter}.wav'
        chunk.export(filename, format="wav")
        counter += 1
        start += threshold
    return counter

if not os.path.isdir("Audiosplit_Result"):
    os.mkdir("Audiosplit_Result")
batas = input('Banyak File Audio .wav yang Akan di Split : ')
path_dir = input('Letak Directory/Folder yang akan digunakan untuk Menyimpan Audio Split : ')
dir_audio = input('Letak Directory Audio .wav yang akan digunakan untuk proses Split : ')
filename = input('Berikan Nama Folder yang akan digunakan :  ')
filename1 = filename
batas = int(batas)
while count <= batas:
    #create_dir(path_dir,filename,count)
    increm = split_audio(path_dir, dir_audio, filename1, count, increm)
    count += 1


#C:/Users/Rizky Zull Fhamy/Documents/Bismillah_TA/Dataset/SplitAudio_5s/Audiosplit_Result