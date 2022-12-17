import os

folder = "C:\\Users\\Hp\\Github\\Age-Recognition-App\\Datasets\\room_street_data\\street_data"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"streetDS{str(count)}.jpg"
    src = f"{folder}/{filename}"
    dst = f"{folder}/{dst}"# foldername/filename, if .py file is outside folder 
    os.rename(src, dst)
    



