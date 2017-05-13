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
    cur_path = line[0]
    # print(cur_path)
    # img = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2GRAY)
    # img = img.reshape(img.shape[0], img.shape[1], 1)
    if not os.path.exists(cur_path):
        continue
    img = cv2.imread(cur_path)
    imgs.append(img)

    img = cv2.flip(img, 1)
    # img = img.reshape(img.shape[0], img.shape[1], 1)
    imgs.append(img)
    mes = float(line[3])
    measurements.append(mes)
    measurements.append(-1.0 * mes)

    # img = cv2.cvtColor(cv2.imread(line[1]), cv2.COLOR_BGR2GRAY)
    # img = img.reshape(img.shape[0], img.shape[1], 1)
    imgs.append(cv2.imread(line[1])) # left
    measurements.append(mes + 0.2)

    # img = cv2.cvtColor(cv2.imread(line[2]), cv2.COLOR_BGR2GRAY)
    # img = img.reshape(img.shape[0], img.shape[1], 1)
    imgs.append(cv2.imread(line[2])) # right
    measurements.append(mes - 0.2)


X_train = np.array(imgs)
y_train = np.array(measurements)
print(X_train.shape, y_train.shape)

def pre_process(img):
    from keras.backend import tf as ktf

    # sess  = K.get_session()
    # img = sess.run(img) # now img is a proper numpy array 

    # # img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV) 
    # # img=cv2.resize(img,(200,66))
    # # return img
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.array(img)
    # return img.reshape((img.shape[0], img.shape[1], 1))
    print(img)
    return ktf.image.rgb_to_grayscale(img)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D

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


model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, epochs=3, shuffle=True)
model.save('model2.h5')
