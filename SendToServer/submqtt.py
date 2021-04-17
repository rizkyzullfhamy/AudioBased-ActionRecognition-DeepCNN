import paho.mqtt.client as mqtt

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  # Subscribing in on_connect() means that if we lose the connection and
  # reconnect then subscriptions will be renewed.
  client.subscribe("/action/aurecog/5f8f2dd0c00456e8e03e5e9c")
  
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    lenPayload = len(msg.payload) 
    if (lenPayload <= 20):
        print("Receive message =", str(msg.payload.decode("utf-8")))
    else:
        f = open("C:/Users/LENOVO/Documents/RizkyZullFhamy/AudioBased-ActionRecognition-DeepCNN/Result_Image/Output.jpg","wb")
        f.write(msg.payload)
        f.close()
        print ("image received")
  
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.emqx.io", 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()
