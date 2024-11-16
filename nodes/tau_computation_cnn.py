#!/usr/bin/env python3
import cv2
from os import listdir
# from sklearn.metrics import confusion_matrix
import rospy
import tensorflow as tf
from tensorflow import keras
import openpyxl
import os
from tkinter import W
from sensor_msgs.msg import LaserScan
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt_ml.msg import TauComputation
from sensor_msgs.msg import Image, LaserScan
from xlsxwriter import Workbook
import autokeras as ak
import pandas as pd
import time

# Definition of the limits for the ROIs
def set_limit(img_width, img_height):
    ########## IMPORTANT PARAMETERS: ##########
	# Extreme left and extreme right
	global x_init_el
	global y_init_el
	global x_end_el
	global y_end_el
	x_init_el = 0
	y_init_el = 0
	x_end_el = int(3 * img_width / 12)
	y_end_el = int(11 * img_height / 12)

	global x_init_er
	global y_init_er
	global x_end_er
	global y_end_er
	x_init_er = int(9 * img_width / 12)
	y_init_er = 0
	x_end_er = int(img_width)
	y_end_er = int(11 * img_height / 12)

	# Left and right
	global x_init_l
	global y_init_l
	global x_end_l
	global y_end_l
	x_init_l = int(3 * img_width / 12)
	y_init_l = int(1 * img_height / 12)
	x_end_l = int(5 * img_width / 12)
	y_end_l = int(9.5 * img_height / 12)

	global x_init_r
	global y_init_r
	global x_end_r
	global y_end_r
	x_init_r = int(7 * img_width / 12)
	y_init_r = int(1 * img_height / 12)
	x_end_r = int(9 * img_width / 12)
	y_end_r = int(9.5 * img_height / 12)
    
    # Centre
	global x_init_c
	global y_init_c
	global x_end_c
	global y_end_c
	x_init_c = int(5.5 * img_width / 12)
	y_init_c = int(2.5 * img_height / 12)
	x_end_c = int(6.5 * img_width / 12)
	y_end_c = int(7.5 * img_height / 12)


# Visual representation of the ROIs with the average TTT values
def draw_image_segmentation(curr_image, tau_el, tau_er, tau_l, tau_r, tau_c):
    color_image = curr_image
    color_blue = [255, 225, 0]  
    color_green = [0, 255, 0]
    color_red = [0, 0, 255]
    linewidth = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extreme left and extreme right
    cv2.rectangle(color_image, (x_init_el, y_init_el), (x_end_el, y_end_el), color_blue, linewidth)
    cv2.rectangle(color_image, (x_init_er, y_init_er), (x_end_er, y_end_er), color_blue, linewidth)
    cv2.putText(color_image, str(round(tau_el, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/2)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(tau_er, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 2)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Left and right
    cv2.rectangle(color_image, (x_init_l, y_init_l), (x_end_l, y_end_l), color_green, linewidth)
    cv2.rectangle(color_image, (x_init_r, y_init_r), (x_end_r, y_end_r), color_green, linewidth)
    cv2.putText(color_image, str(round(tau_l, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 2)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(tau_r, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 2)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Centre 
    cv2.rectangle(color_image, (x_init_c, y_init_c), (x_end_c, y_end_c), color_red, linewidth)
    cv2.putText(color_image, str(round(tau_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 2)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('ROIs Representation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROIs Representation', (600, 600))
    cv2.imshow('ROIs Representation', color_image)
    cv2.waitKey(10)
def draw_image_segmentation_2(curr_image, taup_el, taup_er, taup_l, taup_r, taup_c,tau_el, tau_er, tau_l, tau_r, tau_c):
    color_image = curr_image #cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
    color_blue = [255, 225, 0]  
    color_green = [0, 255, 0]
    color_red = [0, 0, 255]
    linewidth = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extreme left and extreme right
    cv2.rectangle(color_image, (x_init_el, y_init_el), (x_end_el, y_end_el), color_blue, linewidth)
    cv2.rectangle(color_image, (x_init_er, y_init_er), (x_end_er, y_end_er), color_blue, linewidth)
    cv2.putText(color_image, str(round(tau_el, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/2)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(tau_er, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 2)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_el, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/3)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_er, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 3)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Left and right
    cv2.rectangle(color_image, (x_init_l, y_init_l), (x_end_l, y_end_l), color_green, linewidth)
    cv2.rectangle(color_image, (x_init_r, y_init_r), (x_end_r, y_end_r), color_green, linewidth)
    cv2.putText(color_image, str(round(tau_l, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 2)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(tau_r, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 2)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_l, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 3)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_r, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 3)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Centre 
    cv2.rectangle(color_image, (x_init_c, y_init_c), (x_end_c, y_end_c), color_red, linewidth)
    cv2.putText(color_image, str(round(tau_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 2)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(taup_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 3)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('ROIs Representation (top value pred bottom value lidar)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROIs Representation (top value pred bottom value lidar)', (600, 600))
    cv2.imshow('ROIs Representation (top value pred bottom value lidar)', color_image)
    cv2.waitKey(10)

class Calculate_Tau():
    def __init__(self):
        # Tau Publisher
        self.tau_values = rospy.Publisher("tau_values_cnn", TauComputation, queue_size=10)
        self.tau_values_lidar = rospy.Publisher("tau_values_lidar", TauComputation, queue_size=10)
        # Raw Image Subscriber
        self.image_sub_name = "/realsense/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        self.bridge = CvBridge()
        self.curr_image = None
        self.prev_image = None
        path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/trained_model_parameters/"
        
        self.model = tf.keras.models.load_model(path + "renest101v2_model_64_updated.h5", compile=False, custom_objects=ak.CUSTOM_OBJECTS)
        self.model.compile(optimizer = 'adam', loss = 'mae', metrics = ['MeanSquaredError', 'mean_absolute_error']) #Paste it here
        
        # self.interpreter = tf.lite.Interpreter(model_path=str(model_resnet101.tflite))
        # self.interpreter.allocate_tensors()
        
        self.vel = 0.7 # change according to the speed of the jackal specified in the controller node
        # Lidar Subscriber
        self.sub = rospy.Subscriber('/front/scan', LaserScan, self.callback_lidar)
        # self.sub = rospy.Subscriber('/rear/scan', LaserScan, self.callback_lidar)
        self.ranges = None
        self.increments = None
        self.linear_x_vel = 1
        self.angle_min = None
        self.angle_max = None

        # to save data for plotting
        # self.frame = np.array([])
        # self.el_cnn = np.array([])
        # self.er_cnn = np.array([]) 
        # self.l_cnn = np.array([])
        # self.r_cnn = np.array([])
        # self.c_cnn = np.array([])
        # self.el_lidar = np.array([])
        # self.er_lidar =  np.array([])
        # self.l_lidar = np.array([])
        # self.r_lidar = np.array([])
        # self.c_lidar = np.array([])
        # self.count = 0

    def callback_img(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)
            return
        # Get time stamp
        self.secs = data.header.stamp.secs
        self.nsecs = data.header.stamp.nsecs
        self.width = data.width
        self.height = data.height

    def get_tau_values_given_velocity(self):
        # img_size = 200
        
        if self.curr_image is not None:
            if self.prev_image is not None:
                print("here")
                img_size = 250
                curr_image = self.curr_image

                img_1 = cv2.resize(self.prev_image,(img_size,img_size))
                img_2 = cv2.resize(self.curr_image,(img_size,img_size))
               
                img = np.concatenate([img_1, img_2], 2)
                img = tf.expand_dims(img, 0)
                img = np.asarray(img)
                print("img_shape", np.shape(img)
                )
                # print(img.shape)
                vel = 0.7
                vel = np.asarray([vel])
                # print('v',vel.shape)
                start = time.time()
                tau_pred = self.model.predict({"input_1": img, "input_2": vel})
                
                # test_image = img.astype(np.float32)

                # input_index = self.interpreter.get_input_details()[0]["index"]
                # output_index = self.interpreter.get_output_details()[0]["index"]

                # self.interpreter.set_tensor(input_index, {test_image, vel})
                # self.interpreter.invoke()
                # tau_pred = self.interpreter.get_tensor(output_index)
                
                
                
                print(f"func:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
                set_limit(self.width, self.height)

                # Publish Tau values data to rostopic
                # Creation of TauValues.msg
                msg = TauComputation()
                msg.header.stamp.secs =  self.secs
                msg.header.stamp.nsecs =  self.nsecs
                msg.height = self.height
                msg.width = self.width

                msg.tau_el = tau_pred[0][0]
                msg.tau_er = tau_pred[0][4]
                msg.tau_l = tau_pred[0][1]
                msg.tau_r = tau_pred[0][3]
                msg.tau_c = tau_pred[0][2]
                self.tau_values.publish(msg)
                self.prev_image = self.curr_image
                
                #for plotting
                if tau_pred[0][0] < 0:
                    tau_pred[0][0] = np.inf

                if tau_pred[0][1] < 0:
                    tau_pred[0][1] = np.inf

                if tau_pred[0][2] < 0:
                    tau_pred[0][2] = np.inf

                if tau_pred[0][3] < 0:
                    tau_pred[0][3] = np.inf

                if tau_pred[0][4] < 0:
                    tau_pred[0][4] = np.inf
                
                
                draw_image_segmentation_2(curr_image, tau_pred[0][0], tau_pred[0][4], tau_pred[0][1], tau_pred[0][3], tau_pred[0][2], self.tau_el, self.tau_er, self.tau_l, self.tau_r, self.tau_c)
               
                    # print('error to draw')
                # draw_image_segmentation(curr_image,  tau_pred[0][0], tau_pred[0][4], tau_pred[0][1], tau_pred[0][3], tau_pred[0][2])
        
                # save the data for plotting
                
                # self.frame = np.append(self.frame, self.count)
                # self.el_cnn = np.append(self.el_cnn, tau_pred[0][0])
                # self.er_cnn = np.append(self.er_cnn, tau_pred[0][4])
                # self.l_cnn = np.append(self.l_cnn, tau_pred[0][1])
                # self.r_cnn = np.append(self.r_cnn, tau_pred[0][3])
                # self.c_cnn = np.append(self.c_cnn, tau_pred[0][2])
                # self.el_lidar = np.append(self.el_lidar, self.tau_el)
                # self.er_lidar = np.append(self.er_lidar, self.tau_er)  
                # self.l_lidar = np.append(self.l_lidar, self.tau_l) 
                # self.r_lidar = np.append(self.r_lidar, self.tau_r) 
                # self.c_lidar = np.append(self.c_lidar, self.tau_c) 
                # # self.count += 1
            else:  
                self.prev_image = self.curr_image
                
    def get_tau_values_without_velocity(self):
        if self.curr_image is not None:
            if self.prev_image is not None:
                print("here")
                img_size = 250
                curr_image = self.curr_image

                img_1 = cv2.resize(self.prev_image,(img_size,img_size))

                img_2 = cv2.resize(self.curr_image,(img_size,img_size))
               
                img = np.concatenate([img_1, img_2], 2)
                img = tf.expand_dims(img, 0)
                img = np.asarray(img)
                print("img_shape", np.shape(img))
             
                vel = np.asarray([self.vel])

                tau_pred = self.model.predict({"input_1": img})
                set_limit(self.width, self.height)

                # Publish Tau values data to rostopic
                # Creation of TauValues.msg
                msg = TauComputation()
                msg.header.stamp.secs =  self.secs
                msg.header.stamp.nsecs =  self.nsecs
                msg.height = self.height
                msg.width = self.width

                msg.tau_el = tau_pred[0][0]/vel
                msg.tau_er = tau_pred[0][4]/vel
                msg.tau_l = tau_pred[0][1]/vel
                msg.tau_r = tau_pred[0][3]/vel
                msg.tau_c = tau_pred[0][2]/vel
                
                self.tau_values.publish(msg)
                self.prev_image = self.curr_image
                
                draw_image_segmentation(curr_image, tau_pred[0][0], tau_pred[0][4], tau_pred[0][1], tau_pred[0][3], tau_pred[0][2], self.tau_el, self.tau_er, self.tau_l, self.tau_r, self.tau_c)
            
            else:  
                self.prev_image = self.curr_image

    def callback_lidar(self, msg):
        start_ind  = 230 #0
        end_ind = 488 #len(msg.ranges) - 1   #488 #
        # print('ol',msg.angle_max)
        self.angle_min = msg.angle_min + start_ind * msg.angle_increment
        self.angle_max = msg.angle_min + end_ind * msg.angle_increment
        self.increments = msg.angle_increment
        self.ranges = msg.ranges[230:489]
        # print(self.ranges)
        if self.ranges is not None:
            theta_rd = np.arange(self.angle_min,self.angle_max + self.increments, self.increments, dtype=float) # generated within the half-open interval [start, stop).
            # print('rd',theta_rd)
            theta_deg = theta_rd * (180/np.pi)
            ranges = self.ranges
            # print('deg',len(theta_deg))
            # print('ran',len(ranges))
            tau_val = np.array([])
            for i in range(len(ranges)):
                # print('i',i)
                # print('ran',ranges[i]*np.cos(theta_deg[i]))
                tau_val = np.append(tau_val,abs(ranges[i]*np.cos(theta_deg[i])))
            # print(self.angle_min,self.angle_max)
            # print('be',tau_val)
            self.tau_val = tau_val/self.linear_x_vel
        
            # Extreme left
            count_inf_el = 0
            count_el = 0
            tau_val_el = self.tau_val[204:259]
            tau_el = 0
            for i in range(len(tau_val_el)):
                if tau_val_el[i] == np.inf: 
                    count_inf_el += 1
                    if count_inf_el > int(len(tau_val_el)/2):
                        tau_el = np.inf
                        break
                else:
                    tau_el += tau_val_el[i]
                    count_el += 1
            if count_el ==0:
                tau_el = np.inf
            else:
                tau_el = tau_el / count_el

            # Extreme right
            count_inf_er = 0
            count_er = 0
            tau_val_er = tau_val[0:56]
            tau_er = 0       
            for i in range(len(tau_val_er)):
                if tau_val_er[i] == np.inf: 
                    count_inf_er += 1
                    if count_inf_er > int(len(tau_val_er)/2):
                        tau_er = np.inf
                        break
                else: 
                    tau_er += tau_val_er[i]
                    count_er += 1
            if count_er == 0:
                tau_er = np.inf
            else:
                tau_er = tau_er / count_er

            # left
            count_inf_l = 0
            count_l = 0
            tau_val_l = tau_val[154:205]
            tau_l = 0
            for i in range(len(tau_val_l)):
                if tau_val_l[i] == np.inf: 
                    count_inf_l += 1
                    if count_inf_l > int(len(tau_val_l)/2):
                        tau_l = np.inf
                        break
                else: 
                    tau_l += tau_val_l[i]
                    count_l += 1
            if count_l == 0:
                tau_l = np.inf
            else:
                tau_l = tau_l / count_l

            # Right
            count_inf_r = 0
            count_r = 0
            tau_val_r = tau_val[55:103]
            tau_r = 0
            for i in range(len(tau_val_r)):
                if tau_val_r[i] == np.inf: 
                    count_inf_r += 1
                    if count_inf_r > int(len(tau_val_r)/2):
                        tau_r = np.inf
                        break
                else: 
                    tau_r += tau_val_r[i]
                    count_r += 1
            if count_r == 0:
                tau_r = np.inf
            else:
                tau_r = tau_r / count_r

            # Centre
            count_inf_c = 0
            count_c = 0
            tau_val_c = tau_val[117:144]
            tau_c = 0
            for i in range(len(tau_val_c)):
                if tau_val_c[i] == np.inf: 
                    count_inf_c+=1
                    if count_inf_c > int(len(tau_val_c)/2):
                        tau_c = np.inf
                        break
                else: 
                    tau_c += tau_val_c[i]
                    count_c +=1
            if count_c == 0:
                tau_c = np.inf
            else:
                tau_c = tau_c / count_c 
            
            self.tau_el = tau_el
            self.tau_l = tau_l
            self.tau_c = tau_c
            self.tau_r = tau_r
            self.tau_er = tau_er

            msg = TauComputation()
            msg.header.stamp.secs =  self.secs
            msg.header.stamp.nsecs =  self.nsecs
            msg.height = self.height
            msg.width = self.width

            msg.tau_el = tau_el
            msg.tau_er = tau_er
            msg.tau_l = tau_l
            msg.tau_r = tau_r
            msg.tau_c = tau_c
            self.tau_values_lidar.publish(msg)

    def on_rospy_shutdown(self):
        rospy.logwarn("Stopping")
        rospy.Rate(1).sleep()
        
        count = 1
        
        path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/data_for_plotting/cnn_lidar_L_boxes_corridor/" #house_garden
        # folder_name = filter + str(count) + '/'
        # os.mkdir(path_folder + folder_name) 

        # frame_array = pd.DataFrame(self.frame)
        # frame_array.to_excel( path_folder + 'frame_array_' + str(count) + '.xlsx')

        el_cnn_array = pd.DataFrame(self.el_cnn)
        el_cnn_array.to_excel( path_folder + 'el_cnn_array_' + str(count) + '.xlsx')

        er_cnn_array = pd.DataFrame(self.er_cnn)
        er_cnn_array.to_excel( path_folder + 'er_cnn_array_' + str(count) + '.xlsx')

        l_cnn_array = pd.DataFrame(self.l_cnn)
        l_cnn_array.to_excel( path_folder + 'l_cnn_array_' + str(count) + '.xlsx')

        r_cnn_array = pd.DataFrame(self.r_cnn)
        r_cnn_array.to_excel( path_folder + 'r_cnn_array_' + str(count) + '.xlsx')

        c_cnn_array = pd.DataFrame(self.c_cnn)
        c_cnn_array.to_excel( path_folder + 'c_cnn_array_' + str(count) + '.xlsx')

        el_lidar_array = pd.DataFrame(self.el_lidar)
        el_lidar_array.to_excel( path_folder + 'el_lidar_array_' + str(count) + '.xlsx')
        
        er_lidar_array = pd.DataFrame(self.er_lidar)
        er_lidar_array.to_excel( path_folder + 'er_lidar_array_' + str(count) + '.xlsx')

        l_lidar_array = pd.DataFrame(self.l_lidar)
        l_lidar_array.to_excel( path_folder + 'l_lidar_array_' + str(count) + '.xlsx')

        r_lidar_array = pd.DataFrame(self.r_lidar)
        r_lidar_array.to_excel( path_folder + 'r_lidar_array_' + str(count) + '.xlsx')

        c_lidar_array = pd.DataFrame(self.c_lidar)
        c_lidar_array.to_excel( path_folder + 'c_lidar_array_' + str(count) + '.xlsx')


if __name__ == '__main__':
    rospy.init_node('tau_from_cnn', anonymous=True)
    tau = Calculate_Tau()
    # rospy.on_shutdown(tau.on_rospy_shutdown)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        tau.get_tau_values_given_velocity()
        r.sleep()
    
    

