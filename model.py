#""" Udacity - CarND - Behavioral cloning project """

import os
import csv

import cv2
import numpy as np
import sklearn
import sklearn.utils
from sklearn.model_selection import train_test_split

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                
#            """ Load the three camera images and correct measurements """

                for i in range(3):
                    
                    name_split = batch_sample[i].split('\\')                
                    
                    name =  name_split[-3] + '\\' + \
                    name_split[-2] + '\\' + name_split[-1]
                    
                    center_image = cv2.imread(name)
                    b,g,r = cv2.split(center_image)
                    img2 = cv2.merge([r,g,b])
                    if i == 0:
                        center_angle = float(batch_sample[3])
                    if i == 1:
                        center_angle = float(batch_sample[3]) + 0.25            
                    if i == 2:
                        center_angle = float(batch_sample[3]) - 0.25   
                    
                    images.append(img2)
                    angles.append(center_angle)
                    
                    
                    
            """ Flip images to augment data """

            augmented_images, augmented_measurements = [], []   
            
            for image, measurement in zip(images,angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            
            yield sklearn.utils.shuffle(X_train, y_train)


""" Define model
NVidia Pipeline

"""

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization

def cnn_model(input_shape):
    
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
     # trim image to only see section with road
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_normal'))
    model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_normal'))
    model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_normal'))
    model.add(Conv2D(64,(3,3),activation='relu', padding='valid', kernel_initializer='he_normal'))
    model.add(Conv2D(64,(3,3),activation='relu', padding='valid', kernel_initializer='he_normal'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu', kernel_initializer='he_normal'))
    # Dropout layer to avoid overfiting
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, kernel_initializer='he_normal'))
    
    return model            
            

def training_run():
    
    """ Load CSV file """
    
    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    """Instantiate model"""
    
    model = cnn_model((160,320,3))
    
    
    """ 
    The batch size, steps per epoch and the number of epochs
    are decisive for the performance of the model
    
    """    
    batch_size=32
    steps_per_epoch=32
    number_of_epochs=3
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    # Using adam optimizer
    model.compile(loss='mse', optimizer='adam')
    
    """Training process"""
    
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch
                , validation_data=validation_generator, 
                validation_steps=steps_per_epoch, epochs=number_of_epochs)
    
    model.save('model2.h5')
    
    
if __name__ == "__main__":
    training_run()

