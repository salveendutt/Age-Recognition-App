import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sb
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

imagesPath = "../../Face Recognition Models/faceDetected/faces"

img_width = 105
img_height = 105

images = []
ages = []
i = 0


def class_labels_reassign(age):
    if 1 <= age <= 2:
        return 0
    elif 3 <= age <= 9:
        return 1
    elif 10 <= age <= 20:
        return 2
    elif 21 <= age <= 27:
        return 3
    elif 28 <= age <= 34:
        return 4
    elif 35 <= age <= 46:
        return 5
    elif 47 <= age <= 65:
        return 6
    else:
        return 7


oldPath = "../../../Face Recognition Models/facesDetected/faces/"
newPath = "C:/Users/mrsal/Github Repositories/Dataset/Age Recognition/unified"
for image in os.listdir(newPath):
    img = cv2.imread(newPath + "/" + image)
    # img = img/255
    # img = image.img_to_array(img)
    images.append(img)
    ages.append(class_labels_reassign(np.float(image.split("_")[0])))
    i = i + 1


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image to grayscale.
    img = cv2.equalizeHist(img)
    img = img / 255  # normalizing image.
    img = cv2.resize(img, (32, 32))  # resizing it.
    return img


X_train, X_valid, y_train, y_valid = train_test_split(images, ages, test_size=0.33, random_state=42)

ImagesTrainingAfterPreProcessing = []
ImagesValidationAfterPreProcessing = []
for x in X_train:
    ImagesTrainingAfterPreProcessing.append(preprocessing(x))
for x in X_valid:
    ImagesValidationAfterPreProcessing.append(preprocessing(x))
# ages = np.array(ages)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
ImagesTrainingAfterPreProcessing = np.array(ImagesTrainingAfterPreProcessing)
ImagesValidationAfterPreProcessing = np.array(ImagesValidationAfterPreProcessing)
y_validBeforeCategorical = y_valid
y_train = to_categorical(y_train, 8)
y_valid = to_categorical(y_valid, 8)

ImagesTrainingAfterPreProcessing = ImagesTrainingAfterPreProcessing.reshape(ImagesTrainingAfterPreProcessing.shape[0],
                                                                            ImagesTrainingAfterPreProcessing.shape[1],
                                                                            ImagesTrainingAfterPreProcessing.shape[2],
                                                                            1)
ImagesValidationAfterPreProcessing = ImagesValidationAfterPreProcessing.reshape(
    ImagesValidationAfterPreProcessing.shape[0], ImagesValidationAfterPreProcessing.shape[1],
    ImagesValidationAfterPreProcessing.shape[2], 1)

stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model = Sequential()
model.add(
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(rate=0.5))
model.add(Dense(8, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(ImagesTrainingAfterPreProcessing, y_train, epochs=32, verbose=1)
model.save("./CNN_MODEL_2.h5")
predictions = model.predict(ImagesValidationAfterPreProcessing)

# print(accuracy_score(predictions,y_valid))

result = []

for i in range(0, len(y_valid)):
    result.append(np.argmax(predictions[i]))

resultArray = np.array(result)
resultArray.reshape((1, resultArray.shape[0]))
print("Classification Report: \n", classification_report(y_validBeforeCategorical, resultArray))
hotNCold = tf.math.confusion_matrix(y_validBeforeCategorical, resultArray)
# print(hotNCold)
plt.figure(figsize=(60, 60))
sb.set()
heatMap = sb.heatmap(hotNCold, annot=True, fmt='d', cmap="YlGnBu")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()

# import os
# import tensorflow as tf
# from keras import Sequential,Model,models
# from keras.layers import Dense,Dropout,Flatten,BatchNormalization, GlobalAveragePooling2D
# from keras.applications.resnet import ResNet50, preprocess_input
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from keras.preprocessing import image
# import cv2
# from keras.utils.np_utils import to_categorical
#
# from sklearn.model_selection import train_test_split
# def class_labels_reassign(age):
#     if 1 <= age <= 2:
#         return 0
#     elif 3 <= age <= 9:
#         return 1
#     elif 10 <= age <= 20:
#         return 2
#     elif 21 <= age <= 27:
#         return 3
#     elif 28 <= age <= 45:
#         return 4
#     elif 46 <= age <= 65:
#         return 5
#     else:
#         return 6
# imagesPath = "C:/Users/mrsal/Downloads/faces"
# img_width = 105
# img_height = 105
# images = []
# ages = []
# i = 0
# for image in os.listdir("C:/Users/mrsal/Downloads/faces"):
#     img = cv2.imread("C:/Users/mrsal/Downloads/faces/" + image)
#     img = img/255
#     # img = image.img_to_array(img)
#     images.append(img)
#     ages.append(class_labels_reassign(np.float(img.split("_")[0])))
#     if i == 500:
#         break
#     i = i + 1
# X_train, X_valid, y_train, y_valid = train_test_split(images, ages, test_size=0.33, random_state=42)
# resnet = ResNet50(weights="imagenet", include_top=False, classes=7)
# cnt = 0
# for layers in resnet.layers:
#     if cnt < 150:
#         layers.trainable = False
#     cnt += 1
# x = resnet.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# x = Flatten()(x)
# predictions = Dense(7, activation='softmax')(x)
# model = Model(inputs=resnet.inputs, outputs=predictions)
# model.summary()
# y_train = to_categorical(y_train, 7)
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
# history = model.fit(X_train, y_train, batch_size=50, epochs=10)
#
