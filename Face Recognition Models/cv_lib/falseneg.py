import cvlib as cv
import cv2
import numpy as np
import pandas as pd 


def readImage(image_path):
    img = cv2.imread(image_path)
    faces, _ = cv.detect_face(img)
    
    return len(faces)

FalseNegative = 0
TrueNegative = 0

numOfFacesInHousesFound = 0
for i in range(0,100):
    numOfFacesInHousesFound = numOfFacesInHousesFound + readImage('C:\\Users\\Hp\\Github\\Age-Recognition-App\\Datasets\\room_street_data\house_data\\houseDS{0}.jpg'.format(i))

TrueNegative = 5249
FalseNegative = numOfFacesInHousesFound

print("True Negative:", TrueNegative)
print("False Negative:", FalseNegative)
