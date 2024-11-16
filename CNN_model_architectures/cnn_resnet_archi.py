#!/usr/bin/env python3
import tensorflow as tf
import json
import numpy as np
import pandas as pd
import os
import cv2
import openpyxl
import autokeras as ak

X = []
y = []
velocity = []

# path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/"
path = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/"
path_tau =  path + "tau_values/tau_value"   
path_folder = path + "training_images/"

# path_tau =  path + "tau_tr/tau_value"   
# path_folder = path + "train_tr/"

folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
print('fols',folders)
for folder in folders:
    print('fol',folder)
    path_images = path_folder + folder + '/'
    images_in_folder = [f for f in os.listdir(path_images) if f.endswith(".png")]
    img_size = 250
    for idx in range(len(images_in_folder)-1):
        try:
            # load the image
            img_1 = cv2.imread(path_images + images_in_folder[idx])
            img_1 = cv2.resize(img_1,(img_size,img_size))

            img_2 = cv2.imread(path_images + images_in_folder[idx+1])
            img_2 = cv2.resize(img_2,(img_size,img_size))
            
            img = np.concatenate([img_1, img_2], 2)
            
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


path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt/trained_model_parameters/"       

# Load the saved model architecture
with open(path_folder +"model_sim_data_architecture.json", "r") as f:
    model_architecture = f.read()


# Recreate the model from the architecture
model = tf.keras.models.model_from_json(model_architecture)
# Summarize the loaded model.
model.summary()

# # plot graph
model_name = "renest101v2_model"
tf.keras.utils.plot_model(model, model_name + ".png", show_shapes=True)

model.compile(optimizer = 'adam', loss = 'mae')

# Define a TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)


## train the model
model.fit({"input_1": X_train, "input_2": v_train}, y_train, batch_size=64, epochs = 100,  validation_data=({"input_1": X_valid, "input_2": v_valid}, y_valid), callbacks=[tensorboard_callback])
# Save the best performing model to file
model.save(path_folder + model_name + ".h5")

test_loss= model.evaluate({"input_1": X_test, "input_2": v_test}, y_test)
print('test loss: {}'.format(test_loss))