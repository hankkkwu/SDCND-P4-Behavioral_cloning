import csv
import matplotlib.image as mpimg
import numpy as np
import cv2
# Open the csv file
lines = []
with open('../../../opt/challenge_data/driving_log1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
   
# split data into training set and validation set
from sklearn.model_selection import train_test_split
train_data, valid_data = train_test_split(lines, test_size=0.2)
#print(len(train_data))#  = 23400
#print(len(valid_data))#  = 5850

from sklearn.utils import shuffle
def generator(lines, batch_size=32):
    ''' Define a python generator'''
    num_line = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0, num_line, batch_size):
            batch_data = lines[offset:offset+batch_size]
            images = []
            angles = []
            # Extract steering angle data from csv file and read the images from camera
            for line in batch_data:
                for camera in range(3):
                    source_path = line[camera]
                    file_name = source_path.split('/')[-1]
                    current_path = '../../../opt/challenge_data/IMG/' + file_name
                    image = mpimg.imread(current_path)
                    images.append(image)
                    if camera == 0:
                        angle = float(line[3])
                    if camera == 1:
                        angle = float(line[3]) + 0.12
                    else:
                        angle = float(line[3]) - 0.12
                    angles.append(angle)
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    angle_flipped = -angle
                    angles.append(angle_flipped)
            X_train = np.array(images)
            Y_train = np.array(angles)
            yield shuffle(X_train, Y_train)

batch_size = 32
train_generator = generator(train_data, batch_size=batch_size)
valid_generator = generator(valid_data, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
import math

model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))   # Normalize inputs
model.add(Cropping2D(cropping=((35,25), (0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(80, (3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.8))   # dropout regularization
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam')
# model.summary()
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_data) / batch_size),
                   epochs=5, verbose=1, validation_data=valid_generator,
                   validation_steps = math.ceil(len(valid_data) / batch_size))
# model.fit(X_train, Y_train, epochs=3, validation_split=0.2, shuffle=True)

model.save('new_model.h5')
