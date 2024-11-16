#!/usr/bin/env python3
from tkinter import W
import rospy
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt_ml.msg import TauComputation
import cv2
from sensor_msgs.msg import Image, LaserScan 
import json
import os
import pandas as pd
import xlsxwriter
from xlsxwriter import Workbook
from PIL import Image as im
from std_msgs.msg import Int16MultiArray
import csv

# import time module
import time

class collect_data():
    def __init__(self):
        self.tau_values = rospy.Subscriber("/tau_values", TauComputation, self.callback_tau_front)
        self.tau_values_= rospy.Subscriber("/tau_values_rear", TauComputation, self.callback_tau_rear)
        # Raw Image Subscriber
        self.image_sub_name = "/realsense/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        # Initialize Image acquisition
        self.bridge = CvBridge()
        self.get_variables()
        self.count_2 += 1
        self.update_variables()
        self.curr_image = None
        self.path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/test_results_img/"
        vel = '1' # velocity 
        self.folder_name = 'training_images_' + str(self.count_2) + '_v_' + vel + '/'
        # create_folder
        # Start by opening the spreadsheet and selecting the main sheet
        self.path_tau = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/test_results_tau/"
        self.path_images = self.path_folder + self.folder_name
        os.mkdir(self.path_folder + self.folder_name)
        self.tau_val = None   
        # self.pub = rospy.Publisher('chatter', Int16MultiArray, queue_size=10) 
        self.curr_time_array = np.array([])
        self.pub_tau = rospy.Publisher("tau_values_combo", TauComputation, queue_size=10)
    
    def callback_tau_front(self,data):
        if data:  
            # print('data',data)
            # start = time.time()
            self.tau_val_front = [data.tau_el, data.tau_l, data.tau_c, data.tau_r, data.tau_er]
            # show time of execution per iteration
            # print(f"Iteration_tau:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

    def callback_tau_rear(self,data):
        if data:  
            # print('data',data)
            # start = time.time()
            self.tau_val_rear = [data.tau_el, data.tau_l, data.tau_c, data.tau_r, data.tau_er]
            # show time of execution per iteration
            # print(f"Iteration_tau:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
            self.tau_val_ = True

    def callback_tau(self, array1, array2):
        min_value = np.array([min(array1[0], array2[0]), min(array1[1], array2[1]), min(array1[2], array2[2]), min(array1[3], array2[3]), min(array1[4], array2[4])])
        
        msg = TauComputation()
        msg.header.stamp.secs =  self.secs
        msg.header.stamp.nsecs =  self.nsecs
        msg.height = self.height
        msg.width = self.width

        msg.tau_el = min_value[0]
        msg.tau_er = min_value[1]
        msg.tau_l = min_value[2]
        msg.tau_r = min_value[3]
        msg.tau_c = min_value[4]
        self.pub_tau.publish(msg)
        
        return min_value
        
    def save_array(self):
        print('save_array')
        path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/test_results_tau/"
        csv_file_name = 'time_combo.csv'#straight_cor
        with open(path_folder + csv_file_name, mode='w') as file:
            writer = csv.writer(file)
            # print(len(self.x), len(self.y))
            for i in range(len(self.curr_time_array)):
                writer.writerow([self.curr_time_array[i]])#, self.heading[i]])
    
    def collection(self):
        if self.curr_image is not None and self.tau_val_ is not None:
            start = time.time()
            print('running')
            wb_tau =  xlsxwriter.Workbook(self.path_tau + "tau_value" + str(self.count) + ".xlsx")
            wksheet_tau = wb_tau.add_worksheet()
            curr_image = self.curr_image
            tau_val = self.callback_tau(self.tau_val_rear, self.tau_val_front)
            print('tau',tau_val)
            self.curr_time_array = np.append(self.curr_time_array,self.curr_time)
            try: 
            # print(np.type(curr_image))
                self.save_image(self.count, self.path_images, curr_image)
            except:
                return
            
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
            wksheet_tau.write('F1',time.time())
            wb_tau.close()


            print(f"Iteration_saving:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
            self.count += 1
            print(self.count)
            self.update_variables()

            # direction = Int16MultiArray(data = [1]) #(left,right)
            # self.pub.publish(direction)       

    def callback_img(self, data):
        try:
            start = time.time()
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.curr_image = im.fromarray(self.curr_image)
            print(f"Iteration_img:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
            # Get time stamp
            self.secs = data.header.stamp.secs
            self.nsecs = data.header.stamp.nsecs
            self.height = data.height
            self.width = data.width
            self.curr_time = float(self.secs) + float(self.nsecs) * 1e-9
            # frequency = 1.0 / (curr_time - self.prev_time)
        
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

if __name__ == "__main__":
    rospy.init_node("collect_data")
    collect = collect_data()
    r = rospy.Rate(10)
    rospy.on_shutdown(collect.save_array)
    while not rospy.is_shutdown(): 
        collect.collection()
        r.sleep()   
    
