import math
import numpy as np 

def Rms_Audio(data, sr):
    rms_result = math.sqrt(np.mean(data*data))
    rms_result *= 1000
    print("\nRMS RESULT : ", rms_result)
    # energi = np.sum(data.astype(float)**2)
    # energi = 1.0/(2*(data.size)+1)*np.sum(data.astype(float)**2)/sr
    # print(energi)
    # dataarray.append(energi)
    # print(np.mean(dataarray))
    # print(dataarray)
    return rms_result


