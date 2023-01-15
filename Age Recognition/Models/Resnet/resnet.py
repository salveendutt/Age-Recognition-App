import os
import tensorflow as tf
from keras import Sequential,Model,models
from keras.layers import Dense,Dropout,Flatten,BatchNormalization
from keras.applications.resnet import ResNet50, preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
import cv2
from sklearn.model_selection import train_test_split
imagesPath = "../../Face Recognition Models/faceDetected/faces"

img_width = 105
img_height = 105

images = []
ages = []

for image in os.listdir("../../../Face Recognition Models/facesDetected/faces/"):
    img = cv2.imread("../../../Face Recognition Models/facesDetected/faces/" + image)
    img = img/255
    images.append(img)
    ages.append(image.split("_")[0])

images = np.array(images,dtype=np.float64)
ages = np.array(ages,dtype=np.int)

X_train, X_valid, y_train, y_valid = train_test_split(images,ages, test_size=0.33, random_state=42)
resnet = ResNet50(input_shape=(img_width,img_height,3),weights="imagenet",include_top=False)

cnt = 0
for layers in resnet.layers:
    if cnt < 150:
        layers.trainable = False
    cnt += 1
cnt

x = resnet.output
x = Flatten()(x)
x = Dense(1024,activation='relu')
predictions = Dense(1,activation='linear')(x)
model = Model(inputs = resnet.inputs, outputs = predictions)
model.summary()
model.compile(optimizer = "adam",loss="mse",metrics=['mae'])
history = model.fit(X_train,y_train, batch_size=50, validation_data=(X_valid,y_valid),epochs=50)
model.save("./resnetModel")
