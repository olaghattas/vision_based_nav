#!/usr/bin/env python3
import cv2
import numpy as np
from os import listdir
import tensorflow as tf
from tensorflow import keras
import openpyxl
import os
from tkinter import W
import numpy as np
from xlsxwriter import Workbook

class Train_Model():
    def __init__(self):
        pass
    def train_model(self):
        np.random.seed(0)

        X = []
        y = []
        # path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/"
        path = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/"
        path_tau = path + "tau_values/tau_value"   
        path_images_folder = path + "training_images/"

        folders = [file for file in os.listdir(path_images_folder) if os.path.isdir(os.path.join(path_images_folder, file))]
        for folder in folders:
            path_images = path_images_folder + folder + '/'
            images_in_folder = [f for f in listdir(path_images) if f.endswith(".png")]
            img_size = 250
            for idx in range(len(images_in_folder)-1) :
                print('1')
                try:
                    # Load the colored images
                    img_1 = cv2.imread(path_images + images_in_folder[idx])
                    img_1 = cv2.resize(img_1,(img_size,img_size))

                    img_2 = cv2.imread(path_images + images_in_folder[idx+1])
                    img_2 = cv2.resize(img_2,(img_size,img_size))

                    img = np.concatenate([img_1, img_2], 2)
                    
                    # Add image to the dataset
                    X.append(img)

                    # Retrive distances
                    ps = openpyxl.load_workbook(path_tau + str(images_in_folder[idx+1].split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    y.append(np.asarray(tau_values))

                except Exception as inst:
                    print(idx)
                    print(inst)
         
        X = np.asarray(X)
        print('x_shape', np.shape(X))
        y = np.asarray(y)
        print('y_shape', np.shape(y))

        ind = np.arange(len(X))
        np.random.shuffle(ind)

        # # split the data in 60:20:20 for train:valid:test dataset
        train_size = 0.6
        valid_size = 0.2

        train_index = int(len(ind)*train_size)
        valid_index = int(len(ind)*valid_size)

        X_train = X[ind[0:train_index]]
        X_valid = X[ind[train_index:train_index+valid_index]]
        X_test = X[ind[train_index+valid_index:]]

        y_train = y[ind[0:train_index]]
        y_valid = y[ind[train_index:train_index+valid_index]]
        y_test = y[ind[train_index+valid_index:]]

        # Leave the epochs unspecified for an adaptive number of epochs.
        # Initialize the image regressor.
        # # Convolutional Neural Network
        input_1 = keras.layers.Input(shape=(img_size,img_size,6))
        conv1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(input_1)
        pool1 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(pool1)
        pool2 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(16, kernel_size=3, activation='relu')(pool2)
        pool3 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)
        flat = keras.layers.Flatten()(pool3)
        
        hidden1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(flat)
        drop_1 = keras.layers.Dropout(0.25)
        hidden2 = keras.layers.Dense(225, activation='relu',kernel_initializer='he_uniform')(hidden1)
        drop_1 = keras.layers.Dropout(0.25)(hidden2)
        output = keras.layers.Dense(5,kernel_initializer='he_uniform')(drop_1)

        model = keras.models.Model(inputs=[input_1], outputs=output)

        # summarize layers
        print(model.summary())
      
        model_name = "cnn_colored_output_distance_in_each_roi"
        keras.utils.plot_model(model, model_name + ".png")

        model.compile(optimizer = 'adam', loss = 'mae')
        ## train the model
        model.fit({"input_1": X_train}, y_train, batch_size=64, epochs = 100,  validation_data=({"input_1": X_valid}, y_valid))
        # Save the best performing model to file
        model.save(path + 'trained_model_parameters/' + model_name + ".h5")

        # print(model.predict(X_test)[8])
        test_loss = model.evaluate({"input_1": X_test}, y_test)
        print('test loss: {}'.format(test_loss))




if __name__ == '__main__':
    tr = Train_Model()
    tr.train_model()
