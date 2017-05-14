import csv
import cv2
import numpy as np
import os.path

lines = []

with open('./train/driving_log.csv') as excel:
    reader = csv.reader(excel)
    for line in reader:
        lines.append(line)

imgs = []
measurements = []

for line in lines:
    cur_path = line[0] # center image
    img = cv2.imread(cur_path)
    imgs.append(img)

    img = cv2.flip(img, 1) # horizontal flip
    imgs.append(img)

    mes = float(line[3]) # steering angle for the center image
    measurements.append(mes)
    measurements.append(-1.0 * mes) # steering angle for flipped image

    imgs.append(cv2.imread(line[1])) # left camera image
    measurements.append(mes + 0.2) # correction = +0.2

    imgs.append(cv2.imread(line[2])) # right camera image
    measurements.append(mes - 0.2) # correction = -0.2


X_train = np.array(imgs)
y_train = np.array(measurements)

def pre_process(img):
    # In the preprocessing step, all we do is to simply convert the image from RGB to grayscale
    from keras.backend import tf as ktf
    return ktf.image.rgb_to_grayscale(img)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout

## model which is same as given in nVidia paper
model = Sequential()
model.add(Lambda(pre_process, input_shape = (160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Cropping2D(((70,25),(0,0))))
model.add(Conv2D(24, (5,5), strides= (2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, epochs=3, shuffle=True)
model.save('model2.h5')
