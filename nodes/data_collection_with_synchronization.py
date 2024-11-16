#!/usr/bin/env python3
from tkinter import W
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt_ml.msg import TauComputation
import cv2
from sensor_msgs.msg import Image 
import json
import os
import pandas as pd
import xlsxwriter
# import time module
from PIL import Image as im
import time
import message_filters 

class collect_data():

    def __init__(self):
        # Raw Image Subscriber
        self.image_sub_name = "/realsense/color/image_raw"
        self.tau_sub = message_filters.Subscriber("/tau_values", TauComputation)
        self.image_sub = message_filters.Subscriber(self.image_sub_name, Image)

        ts = message_filters.TimeSynchronizer([self.tau_sub, self.image_sub], 10)
        ts.registerCallback(self.callback)

        # Initialize Image acquisition
        self.bridge = CvBridge()
        self.get_variables()
        self.count_2 += 1
        self.update_variables()

        self.curr_image = None
        self.path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/training_images_trial/"
        vel = '0.5' # velocity 
        self.folder_name = 'training_images_' + str(self.count_2) + '_v_' + vel + '/'
        # create_folder
        # Start by opening the spreadsheet and selecting the main sheet
        self.path_tau = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/tau_values_trial/"
        self.path_images = self.path_folder + self.folder_name
        os.mkdir(self.path_folder + self.folder_name)
        self.tau_val = None   

    def callback(self, tau_sub, image_sub):
        # print('here')
        # print('pp',laser_sub)
        # print('iooo',image_sub)
        self.callback_tau(tau_sub)
        self.callback_img(image_sub)

    def callback_tau(self,data): 
        start = time.time()
        self.tau_val = [data.tau_el, data.tau_l, data.tau_c, data.tau_r, data.tau_er]
        # show time of execution per iteration
        print(f"Iteration_tau:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

    def save_tau_wksheet(self, tau_val):
        wb_tau =  xlsxwriter.Workbook(self.path_tau + "tau_value" + str(self.count) + ".xlsx")
        wksheet_tau = wb_tau.add_worksheet()

        inf = -1 

        try:
                wksheet_tau.write('A1',tau_val[0])
        except:
            wksheet_tau.write('A1',inf)
        try:
            wksheet_tau.write('B1',tau_val[1])
        except:
            wksheet_tau.write('B1',inf)
        try:
            wksheet_tau.write('C1',tau_val[2])
        except:
            wksheet_tau.write('C1',inf)
        try:
            wksheet_tau.write('D1',tau_val[3])
        except:
            wksheet_tau.write('D1',inf)
        try:
            wksheet_tau.write('E1',tau_val[4])
        except:
            wksheet_tau.write('E1',inf)
        wb_tau.close()    

    def callback_img(self, data):
        try:
            start = time.time()
            print("imagesfun")
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.curr_image = im.fromarray(self.curr_image)
            print(f"Iteration_img:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
        except CvBridgeError as e:
            print(e)
            return

    def save_image(self, count : int, shared_path, curr_image): 
            img_name= str(count) + '.png'
            path = shared_path + img_name
            curr_image.save(path)

    def get_variables(self):
        # print("get_variables")
        path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/"
        file = open(path + "variables.json")
        data = json.load(file)
        file.close()
        self.count = data["count"]
        self.count_2 = data["count_2"]
        
    def update_variables(self):
        # print("update_variables")
        path = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/"
        file = open(path + "variables.json", "w")
        updated_data = {"count": self.count, "count_2": self.count_2}
        json.dump(updated_data, file)
        file.close()
    
    def collect_images(self):
        if self.tau_val and self.curr_image:
            start = time.time()
            
            curr_image = self.curr_image
            tau_val = self.tau_val
            self.save_image(self.count, self.path_images,curr_image)
            self.save_tau_wksheet(tau_val)
           
            print(f"Iteration_saving:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
            self.count += 1
            print(self.count)
            self.update_variables()

if __name__ == "__main__":
    rospy.init_node("collect_data")
    collect = collect_data()
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        # collect.collection()
        r.sleep()   
    
