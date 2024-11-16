#!/usr/bin/env python3
from tkinter import W
import rospy
import sensor_msgs.msg
from sensor_msgs.msg import LaserScan
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt_ml.msg import TauComputation
from cv_bridge import CvBridgeError, CvBridge
import cv2
from sensor_msgs.msg import Image
import os
import pandas as pd
import xlsxwriter
from xlsxwriter import Workbook
import time

# el[434,488]
# l ranges[384,434]
# er [230,285]
#r [285,332]
#c [347,373]

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

    color_image = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
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

class compute_tau():
    def __init__(self):
        # Lidar Subscriber
        self.sub = rospy.Subscriber('/front/scan', LaserScan, self.callback)
        # Tau Publisher
        self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=10)
        # Raw Image Subscriber
        self.image_sub_name = "/realsense/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, callback=self.callback_img,queue_size=1,buff_size=2**18) #the buff_size=2**18 avoids delays due to the queue buffer being too small for images
        # self.pub = rospy.Publisher('image_repeated', Image, queue_size=1) 
        
        self.ranges = None
        self.increments = None
        self.linear_x_vel = 1
        self.angle_min = None
        self.angle_max = None
        # Initialize Image acquisition
        self.bridge = CvBridge()
        # self.get_variables()
        self.curr_image = None
        import time

    def callback_img(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)
            return
        # Get time stamp
        self.secs = data.header.stamp.secs
        self.nsecs = data.header.stamp.nsecs
        self.width = data.width
        self.height = data.height
        
    def callback(self, msg):
            start_ind  = 230 #0
            end_ind = 488 #len(msg.ranges) - 1   #488 #
            # print('ol',msg.angle_max)
            self.angle_min = msg.angle_min + start_ind * msg.angle_increment
            self.angle_max = msg.angle_min + end_ind * msg.angle_increment
            self.increments = msg.angle_increment
            self.ranges = msg.ranges[230:489]
            # print(self.ranges)

    def get_tau_values(self):
        start = time.time()
        if self.curr_image is not None:
            curr_image = self.curr_image
            if self.ranges is not None:
                # print(1)
                theta_rd = np.arange(self.angle_min,self.angle_max + self.increments, self.increments, dtype=float) # generated within the half-open interval [start, stop).
                # print('rd',theta_rd)
                theta_deg = theta_rd * (180/np.pi)
                ranges = self.ranges
                # print('deg',len(theta_deg))
                # print('ran',len(ranges))
                tau_val = np.array([])
                for i in range(len(ranges)):
                    tau_val = np.append(tau_val,abs(ranges[i]*np.cos(theta_deg[i])))
                self.tau_val = tau_val/self.linear_x_vel
                
                # inf values causing problems so removing them if number >50%
                # Extreme left
                count_inf_el = 0
                count_el = 0
                tau_val_el = self.tau_val[204:259]
                tau_el = 0
                print('el',tau_val_el)
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
                print('er',tau_val_er)   
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
                print('l',tau_val_l)
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
                print('r',tau_val_r)
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
                print('c',tau_val_c)
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

                print('el',tau_el)
                print('er',tau_er)
                print('l',tau_l)
                print('r',tau_r)
                print('c',tau_c)
                set_limit(self.width, self.height)
                
                # Publish Tau values data to rostopic
                # Creation of TauValues.msg
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
                self.tau_values.publish(msg)

                # Draw the ROIs with their TTT values
                draw_image_segmentation(curr_image, tau_el, tau_er, tau_l, tau_r, tau_c)
        print(f"func:\tTime taken: {(time.time()-start)*10**3:.03f}ms")


if __name__ == '__main__':
    rospy.init_node('tau_from_lidar', anonymous=True)
    val = compute_tau()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        val.get_tau_values()
        r.sleep()