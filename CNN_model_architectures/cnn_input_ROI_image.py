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
from xlsxwriter import Workbook

class train():
    def __init__(self):
        pass
    def train_model(self):
        np.random.seed(0)
        width = 150 # img.shape[1] 
        height = 150 # img.shape[0]

        X = []
        y_1 = [] # tau values 
        y_2 = [] # validity 
        velocity = []

        path = os.environ["HOME"]+"/catkin_ws/src/" # change this according to the location of the folder on your device
        path_tau = path + "vision_based_navigation_ttt/tau_values/tau_value"   
        path_folder = path + "vision_based_navigation_ttt/training_images/"

        folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
        for folder in folders:
            path_images = path + "vision_based_navigation_ttt/training_images/" + folder + '/'
            images_in_folder = [f for f in listdir(path_images) if f.endswith(".png")]

            for idx in range(len(images_in_folder)-1) : 
                print(images_in_folder[idx])
                try:
                    # load the image
                    img_1 = cv2.imread(path_images + images_in_folder[idx],0)
                    img_1 = cv2.resize(img_1,(150,150))

                    img_el_1 = img_1[0 : int((11/12)*height), int((0/12)*width) : int((2.5/12)*width) ] 
                    img_l_1 = img_1[0 : int((11/12)*height), int((2.5/12)*width) : int((5/12)*width) ] 
                    img_c_1 = img_1[0 : int((11/12)*height), int((4.5/12)*width) : int((7/12)*width) ] 
                    img_r_1 = img_1[0 : int((11/12)*height), int((8.5/12)*width) : int((11/12)*width) ] 
                    img_er_1 = img_1[0 : int((11/12)*height), int((9.4/12)*width) : int((11.9/12)*width) ]

                    img_2 = cv2.imread(path_images + images_in_folder[idx+1],0)
                    img_2 = cv2.resize(img_2,(150,150))

                    img_el_2 = img_2[0 : int((11/12)*height), int((0/12)*width) : int((2.5/12)*width) ] 
                    img_l_2 = img_2[0 : int((11/12)*height), int((2.5/12)*width) : int((5/12)*width) ] 
                    img_c_2 = img_2[0 : int((11/12)*height), int((4.5/12)*width) : int((7/12)*width) ] 
                    img_r_2 = img_2[0 : int((11/12)*height), int((8.5/12)*width) : int((11/12)*width) ] 
                    img_er_2 = img_2[0 : int((11/12)*height), int((9.4/12)*width) : int((11.9/12)*width) ] 
                  
                    img_el = np.stack([img_el_1, img_el_2], 2)
                    img_l = np.stack([img_l_1, img_l_2], 2)
                    img_c = np.stack([img_c_1, img_c_2], 2)
                    img_r = np.stack([img_r_1, img_r_2], 2)
                    img_er = np.stack([img_er_1, img_er_2], 2)
                
                    # add image to the dataset
                    X.append(img_el)
                    X.append(img_l)
                    X.append(img_c)
                    X.append(img_r)
                    X.append(img_er)

                    # get velocity
                    vel = float(folder.split('_')[4])
                    velocity.append([vel])
                    velocity.append([vel])
                    velocity.append([vel])
                    velocity.append([vel])
                    velocity.append([vel])

                    # retrive the tau values from the filename
                    ps = openpyxl.load_workbook(path_tau + str(images_in_folder[idx+1].split('.')[0]) + '.xlsx')
                    sheet = ps['Sheet1']
                    #labeled in the following order [el,l,c,r,er]
                    tau_values = [sheet['A1'].value, sheet['B1'].value,sheet['C1'].value,sheet['D1'].value,sheet['E1'].value]
                    
                    y_1.append(np.asarray(tau_values[0])/vel)
                    y_1.append(np.asarray(tau_values[1])/vel)
                    y_1.append(np.asarray(tau_values[2])/vel)
                    y_1.append(np.asarray(tau_values[3])/vel)
                    y_1.append(np.asarray(tau_values[4])/vel)

                    for i in tau_values:
                        if i == -1:
                            y_2.append(0)
                        else:
                            y_2.append(1)

                except Exception as inst:
                    print(idx)
                    print(inst)
         
        X = np.asarray(X)
        # print('x',X.shape)
        y_1 = np.asarray(y_1)
        # print(y_1)
        # print('y',y.shape)
        y_2 = np.asarray(y_2)
        # print(y_2)
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
        input_1 = keras.layers.Input(shape=(137,31,2))
        conv1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(input_1)
        pool1 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(pool1)
        pool2 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(16, kernel_size=3, activation='relu')(pool2)
        pool3 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)
        flat = keras.layers.Flatten()(pool3)
        hidden_1 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(flat)
        output_2 = keras.layers.Dense(1, activation= 'sigmoid', name = 'output_2')(hidden_1)
        input_2 = keras.layers.Input(shape=(1))

        # merge input models
        merge = keras.layers.concatenate([flat, input_2])  

        hidden1 = keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(merge)
        
        hidden2 = keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(hidden1)
        drop_2 = keras.layers.Dropout(0.25)(hidden2)
        output_1 = keras.layers.Dense(1, name = 'output_1')(drop_2)
        

        model = keras.models.Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

        # # summarize layers
        print(model.summary())
        # # plot graph
        keras.utils.plot_model(model, "model_ROI.png", show_shapes=True)
        
        model.compile(optimizer = 'adam', loss = {"output_1" :'mae', "output_2" :'binary_crossentropy'})
        epochs = 100
        ## train the model
        model.fit({"input_1": X_train, "input_2": v_train}, {"output_1": y_1_train, "output_2": y_2_train}, batch_size=64, epochs = epochs,  validation_data=({"input_1": X_valid, "input_2": v_valid}, {"output_1": y_1_valid, "output_2": y_2_valid}))
        with open('model_input_2_output_2___.pkl', 'wb') as files: pickle.dump(model, files)
        # make predictions on test sets
        yhat = model.predict({"input_1": X_test, "input_2": v_test})
        yhat_class = yhat[1].round()

        # calculate accuracy
        acc = accuracy_score(y_2_test, yhat_class)
        print('accuracy_score ','> %.3f' % acc)

        # evaluate model on test set
        mae_1 = mean_absolute_error(y_1_test, yhat[0])
        mae_2 = mean_squared_error(y_1_test, yhat[0])
        print('mean_absolute_error', mae_1, 'mean_squared_error', mae_2)
      
if __name__ == '__main__':
    tr = train()
    tr.train_()


