#!/usr/bin/env python3

# set the matplotlib backend so figures can be saved in the background
# import matplotlib
# matplotlib.use("Agg")

import tensorflow as tf
import cv2
import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
X = []
y = []

def read_from_csv( csv_filename):
        dict_from_csv = {}
        with open(csv_filename, mode='r') as inp:
            reader = csv.reader(inp)
            dict_from_csv = {rows[0]:rows[1] for rows in reader}
        return dict_from_csv

def model_resnet50():
    input_1 = tf.keras.layers.Input(shape=(250,250,3))

    model =  tf.keras.applications.ResNet50(weights=None , include_top=False, input_tensor=input_1)
    # Add prefix to the layer names of the first model
    # for layer in model.layers:
    #     # layer.name = 'model1_' + layer.name
    #     print('$$$$$$$$$$$$$$$$$',layer.name)

    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    output_1 = tf.keras.layers.Dense(3, activation='sigmoid', name='output_1')(x)

    model = tf.keras.models.Model(inputs=[input_1], outputs=[output_1])
    return model

def model_vgg19():
    input_1 = tf.keras.layers.Input(shape=(250,250,3))

    model =  tf.keras.applications.VGG19(weights=None , include_top=False, input_tensor=input_1)
    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    output_1 = tf.keras.layers.Dense(3, activation='sigmoid', name='output_1')(x)
    model = tf.keras.models.Model(inputs=[input_1], outputs=[output_1])
    return model

def alexnet():
    input_1 = tf.keras.layers.Input(shape=(250,250,3))
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])
    return model

def lenet():
    input_1 = tf.keras.layers.Input(shape=(250,250,3))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(6, 5, activation='tanh', input_shape=input_1))
    model.add(tf.keras.layers.AveragePooling2D(2))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(tf.keras.layers.Conv2D(16, 5, activation='tanh'))
    model.add(tf.keras.layers.AveragePooling2D(2))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(tf.keras.layers.Conv2D(120, 5, activation='tanh'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
    return model

path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt/"
# path = os.environ["HOME"]+"/ROS-Jackal/robots/jackal/vision_based_navigation_ttt/" 
path_folder = path + "shape_label_train/"
csv_file_name = 'labels_file.csv'
if os.stat(path + csv_file_name).st_size == 0:
    dict_labels = {}
else:
    dict_labels = read_from_csv(path + csv_file_name)

folders = [file for file in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, file))]
print('fols',folders)
for folder in folders:
    print('fol',folder)
    path_images = path_folder + folder + '/'
    images_in_folder = [f for f in os.listdir(path_images) if f.endswith(".png")]
    img_size = 250
    for idx in range(len(images_in_folder) -1):#
        try:
            # load the image
            img_1 = cv2.imread(path_images + images_in_folder[idx])
            img_1 = cv2.resize(img_1,(img_size,img_size))
           
            # # Convert the image to grayscale
            # gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

            # # Apply Gaussian Blur to reduce noise
            # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # # Perform Canny Edge Detection
            # edges = cv2.Canny(blurred, 50, 150)
            # print(np.shape(edges))
            # edges = np.reshape(edges,(img_size,img_size,1))
            # # add image to the dataset
            # print(np.shape(edges))
            # img = np.concatenate([img_1,edges],2)
            X.append(img_1)
            # print(np.shape(edges))
            
            # retreive direction
            label = dict_labels[str(images_in_folder[idx].split('.')[0])]
            # print('label',type(eval(label)))
            y.append(eval(label))
          
        except Exception as inst:
            print(idx)
            print(inst)

print(np.shape(X))
print(np.shape(y))
X = np.asarray(X)
y = np.asarray(y)
epochs = 100
# y_2 = np.asarray(y_2)
# y_3 = np.asarray(y_3)

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


# model = model_resnet50()
model = model_vgg19()

root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

# summarize layers
# print(model.summary())

# # plot graph
model_name = "determine_shape_model"
tf.keras.utils.plot_model(model, model_name + ".png", show_shapes=True)

model.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])

## train the model
history = model.fit({"input_1": X_train}, y_train, batch_size=32, epochs = epochs,  validation_data=({"input_1": X_valid}, y_valid), callbacks=[tensorboard_cb])
# Save the best performing model to file
model.save(path_folder + model_name + ".h5")

loss, recall, accuracy, precision= model.evaluate({"input_1": X_test},y_test)
# # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
# # print('test loss: {}'.format(test_loss))
f1 = 2 * (precision * recall) / (precision + recall)

print("F1-score:", f1)
print(precision, recall)
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Training Loss")
plt.legend(loc="upper left")
plt.savefig("lossvsephochs.png")
plt.show()

plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.title("Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Validation Loss")
plt.legend(loc="upper left")
plt.savefig("val_lossvsephochs.png")
plt.show()

plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Training Accuracy")
plt.legend(loc="upper left")
plt.savefig("accvsephochs.png")
plt.show()

plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("valaccvsephochs.png")
plt.show()

