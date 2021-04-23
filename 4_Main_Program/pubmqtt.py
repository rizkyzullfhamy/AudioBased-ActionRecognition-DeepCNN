import paho.mqtt.publish as publish
import time

# dir image = "C:/Users/LENOVO/Documents/RizkyZullFhamy/AudioBased-ActionRecognition-DeepCNN/Result_Image/imagefeature.png"
#Broker = 'broker.emqx.io' atau 103.106.72.187
#Topic  = /action/aurecog/5f8f2dd0c00456e8e03e5e9c

def publish_MQTT(brokerMqtt, TopicMqtt, dir_image, label):
    f= open(dir_image,"rb")
    filecontent = f.read()
    byteArr = bytearray(filecontent)

    publish.single(TopicMqtt, byteArr, qos=1, hostname=brokerMqtt)
    time.sleep(2)
    publish.single(TopicMqtt, label, qos=1, hostname=brokerMqtt)
    
