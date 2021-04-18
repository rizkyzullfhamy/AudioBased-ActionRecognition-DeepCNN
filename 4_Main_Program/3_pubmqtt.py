import paho.mqtt.publish as publish
import time

f= open("C:/Users/LENOVO/Documents/RizkyZullFhamy/AudioBased-ActionRecognition-DeepCNN/Result_Image/imagefeature.png","rb")
filecontent = f.read()
byteArr = bytearray(filecontent)

publish.single('/action/aurecog/5f8f2dd0c00456e8e03e5e9c', byteArr, qos=1, hostname='broker.emqx.io')
time.sleep(4)
publish.single('/action/aurecog/5f8f2dd0c00456e8e03e5e9c', "ListeningMusic", qos=1, hostname='broker.emqx.io')
#103.106.72.187
