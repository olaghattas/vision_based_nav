#!/usr/bin/env python3
import cv2
import numpy as np
from os import listdir

import tensorflow as tf
from tensorflow import keras
import openpyxl
from PIL import Image
import matplotlib.pyplot as plt
import os
from tkinter import W
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt.msg import TauComputation
import pandas as pd
from xlsxwriter import Workbook

class Train():
    def __init__(self):
        pass
    def train_(modelself):
        
        X = []
        y = []
        velocity = []

        # path = os.environ["HOME"]+"/catkin_ws/src/"
        path = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/"
        path_tau =  path + "vision_based_navigation_ttt/tau_values_no_flag/tau_value"   
        path_folder = path + "vision_based_navigation_ttt/training_images/"

        folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
        for folder in folders:

            # print('fol',folder)
            path_images = path + "vision_based_navigation_ttt/training_images/" + folder + '/'
            images_in_folder = [f for f in listdir(path_images) if f.endswith(".png")]
            img_size = 250
            for idx in range(len(images_in_folder)-1):
                try:
                    # load the image
                    img_1 = cv2.imread(path_images + images_in_folder[idx],0)
                    img_1 = cv2.resize(img_1,(img_size,img_size))

                    img_2 = cv2.imread(path_images + images_in_folder[idx+1],0)
                    img_2 = cv2.resize(img_2,(img_size,img_size))
                    
                    img = np.stack([img_1, img_2], 2)
                    
                    # add image to the dataset
                    X.append(img)

                    # get velocity
                    vel = float(folder.split('_')[4])
                    velocity.append([vel])

                    # retrive distances
                    ps = openpyxl.load_workbook(path_tau + str(images_in_folder[idx+1].split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    y.append(np.asarray(tau_values)/vel)

                except Exception as inst:
                    print(idx)
                    print(inst)
         
        X = np.asarray(X)
        y = np.asarray(y)
        velocity = np.asarray(velocity)
        
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

        v_train = velocity[ind[0:train_index]]
        v_valid = velocity[ind[train_index:train_index+valid_index]]
        v_test = velocity[ind[train_index+valid_index:]]

        # # Convolutional Neural Network
        input_1 = keras.layers.Input(shape=(img_size,img_size,2))
        conv1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(input_1)
        pool1 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(pool1)
        pool2 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(16, kernel_size=3, activation='relu')(pool2)
        pool3 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)
        flat = keras.layers.Flatten()(pool3)
        input_2 = keras.layers.Input(shape=(1))

        # merge input models
        merge = keras.layers.concatenate([flat, input_2])  

        hidden1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(merge)
        drop_1 = keras.layers.Dropout(0.25)
        hidden2 = keras.layers.Dense(225, activation='relu',kernel_initializer='he_uniform')(hidden1)
        drop_1 = keras.layers.Dropout(0.25)(hidden2)
        output = keras.layers.Dense(5,kernel_initializer='he_uniform')(drop_1)

        model = keras.models.Model(inputs=[input_1, input_2], outputs=output)

        # # summarize layers
        print(model.summary())
        # # plot graph
        model_name = "model_with_shape_info"
        keras.utils.plot_model(model, model_name + ".png")

        model.compile(optimizer = 'adam', loss = 'mae', metrics = 'accuracy')
      
        ## train the model
        model.fit({"input_1": X_train, "input_2": v_train}, y_train, batch_size=64, epochs = 100,  validation_data=({"input_1": X_valid, "input_2": v_valid}, y_valid))
        # Save the best performing model to file
        model.save(model_name + ".h5")

        # print(model.predict(X_test)[8])
        test_loss, test_acc = model.evaluate({"input_1": X_test, "input_2": v_test}, y_test)
        print('test loss: {}, test accuracy: {}'.format(test_loss, test_acc) )

if __name__ == '__main__':
    tr =  Train()
    tr.train_()
    

