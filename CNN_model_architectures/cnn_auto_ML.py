#!/usr/bin/env python3
import cv2
import numpy as np
from os import listdir
import tensorflow as tf
from tensorflow import keras
import openpyxl
import os
from tkinter import W
from xlsxwriter import Workbook
import autokeras as ak

class Train_Model():
    def __init__(self):
        pass
    def train_model(self):
        X = []
        y = []
        # path = os.environ["HOME"]+"/catkin_ws/src/"
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
        reg = ak.ImageRegressor(max_trials=1, loss='mean_absolute_error')
        # Feed the image regressor with training data.
        reg.fit(X_train, y_train, validation_data=(X_valid, y_valid))

        # Predict with the best model.
        predicted_y = reg.predict(X_test)
        print(predicted_y)

        # Evaluate the best model with testing data. 
        print(reg.evaluate(X_test, y_test)) 
        model_name = "auto_ml_updated_data"
        
        # Get the best performing model.
        model = reg.export_model()
        keras.utils.plot_model(model, model_name + ".png")
        # Summarize the loaded model.
        model.summary()
        # Save the best performing model in hdf5 format
        model.save(path + 'trained_model_parameters/' + model_name + ".h5")

if __name__ == '__main__':
    tr = Train_Model()
    tr.train_model()
 
    

