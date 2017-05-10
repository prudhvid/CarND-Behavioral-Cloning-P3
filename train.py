import csv
import cv2
import numpy as np


lines = []

with open('./train/driving_log.csv') as excel:
    reader = csv.reader(excel)
    for line in reader:
        lines.append(line)

imgs = []
measurements = []

for line in lines:
    cur_path = line[0]
    # print(cur_path)
    img = cv2.imread(cur_path)
    imgs.append(img)
    imgs.append(cv2.flip(img, 1))
    mes = float(line[3])
    measurements.append(mes)
    measurements.append(-1.0 * mes)

X_train = np.array(imgs)
y_train = np.array(measurements)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(((70,25),(0,0))))
model.add(Conv2D(24, (5,5), strides= (2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Flatten())


model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, epochs=3, shuffle=True)
model.save('model.h5')
