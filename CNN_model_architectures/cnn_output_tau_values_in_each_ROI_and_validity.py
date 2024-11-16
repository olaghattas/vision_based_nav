#!/usr/bin/env python3
import cv2
import numpy as np
from os import listdir
import tensorflow as tf
from tensorflow import keras
import openpyxl
from PIL import Image
import os
from tkinter import W
import numpy as np
from xlsxwriter import Workbook

class Train():
    def __init__(self):
        pass
    def train_(self):
        np.random.seed(0)
        img_size = 250
        X = []
        y_1 = []
        y_2 = []
        velocity = []

        path = os.environ["HOME"]+"/catkin_ws/src/"
        path_tau = path + "vision_based_navigation_ttt/tau_values/tau_value"   
        path_folder = path + "vision_based_navigation_ttt/training_images/"

        folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
        for folder in folders:
            path_images = path + "vision_based_navigation_ttt/training_images/" + folder + '/'
            images_in_folder = [f for f in listdir(path_images) if f.endswith(".png")]

            for idx in range(1) : #len(images_in_folder)-1
                # print(images_in_folder[idx])
                try:
                    # load the image
                    img_1 = cv2.imread(path_images + images_in_folder[idx],0)
                    img_1 = cv2.resize(img_1,(img_size,img_size))

                    img_2 = cv2.imread(path_images + images_in_folder[idx+1],0)
                    img_2 = cv2.resize(img_2,(img_size,img_size))
                    img = np.stack([img_1, img_2], 2)
                    
                    # print(img)
                    # add image to the dataset
                    print(img)
                    X.append(img)
                    print(X)

                    # get velocity
                    vel = float(folder.split('_')[4])
                    # print('ve',vel)
                    velocity.append([vel])

                    # retrive the direction from the filename
                    ps = openpyxl.load_workbook(path_tau + str(images_in_folder[idx+1].split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    tau_values = [sheet['A1'].value,sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    y_1.append(np.asarray(tau_values)/vel)

                    y_temp = []
                    for i in tau_values:
                        if i == -1:
                            y_temp.append(0)
                        else:
                            y_temp.append(1)
                    y_2.append(y_temp)

                except Exception as inst:
                    print(idx)
                    print(inst)
         
        X = np.asarray(X)
        y_1 = np.asarray(y_1)
        y_2 = np.asarray(y_2)
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

        y_1_train = y_1[ind[0:train_index]]
        y_1_valid = y_1[ind[train_index:train_index+valid_index]]
        y_1_test = y_1[ind[train_index+valid_index:]]

        y_2_train = y_2[ind[0:train_index]]
        y_2_valid = y_2[ind[train_index:train_index+valid_index]]
        y_2_test = y_2[ind[train_index+valid_index:]]

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
        hidden_1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(flat)
        output_2 = keras.layers.Dense(5, activation= 'sigmoid', name = 'output_2')(hidden_1)
        input_2 = keras.layers.Input(shape=(1))

        # merge input models
        merge = keras.layers.concatenate([flat, input_2])  

        hidden1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(merge)
        
        hidden2 = keras.layers.Dense(225, activation='relu',kernel_initializer='he_uniform')(hidden1)
        drop_2 = keras.layers.Dropout(0.25)(hidden2)
        output_1 = keras.layers.Dense(5, name = 'output_1')(drop_2)

        model = keras.models.Model(inputs=[input_1, input_2], outputs=[output_1, output_2])
        model_name = "trained_model"
        keras.utils.plot_model(model, model_name + ".png")

        # Summarize the loaded model.
        print(model.summary())
        
        model.compile(optimizer = 'adam', loss = {"dense_2" :'mae', "dense_3" :'mae'}, metrics = [['mean_absolute_error'], ['mean_absolute_error']])
        epochs = 100
        ## train the model
        model.fit({"input_1": X_train, "input_2": v_train}, {"dense_2": y_1_train, "dense_3": y_2_train}, batch_size=64, epochs = epochs,  validation_data=({"input_1": X_valid, "input_2": v_valid}, {"dense_2": y_1_valid, "dense_3": y_2_valid}))
        # Save the best performing model to file
        model.save(model_name + ".h5")



if __name__ == '__main__':
    tr = Train()
    tr.train_()
  
    

