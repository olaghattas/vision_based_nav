#!/usr/bin/env python3
from tkinter import W
import rospy
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
import cv2
import json
import os
import xlsxwriter
from PIL import Image as im
import time
import message_filters 
from sensor_msgs.msg import Image, PointCloud2,PointField,LaserScan
from sensor_msgs import point_cloud2
from itertools import chain
from numpy import arctan2, sqrt
import numexpr as ne

class collect_data():
    def __init__(self):
        # Raw Image Subscriber
        self.image_sub_name = "/camera/color/image_raw"
        self.velodyne_sub_name = "/velodyne_points"
        self.velodyne_sub = message_filters.Subscriber(self.velodyne_sub_name, PointCloud2)
        self.image_sub = message_filters.Subscriber(self.image_sub_name, Image)

        ts = message_filters.ApproximateTimeSynchronizer([self.velodyne_sub, self.image_sub], 10, slop = 0.2)
        ts.registerCallback(self.callback)

        # Initialize Image acquisition
        self.bridge = CvBridge()
        self.get_variables()
        self.count_2 += 1
        self.update_variables()
        self.curr_image = None
        self.curr_image_bool = False
        self.path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/"
        vel = '0.5' # velocity 
        self.folder_name = 'training_images_real_data/training_images_' + str(self.count_2) + '_v_' + vel + '/'
        # create_folder
        # Start by opening the spreadsheet and selecting the main sheet
        self.path_tau = self.path_folder + "tau_values_real_data/"
        self.path_images = self.path_folder + self.folder_name
        os.mkdir(self.path_folder + self.folder_name)
        self.tau_val = None   
        self.tau_val_bool = False 

    def callback(self, tau_sub, image_sub):
        print('tau_img')
        # print('pp',laser_sub)
        # print('iooo',image_sub)
        self.callback_tau(tau_sub)
        self.callback_img(image_sub)
  
    def cart2sph(self,x,y,z, ceval=ne.evaluate):
        """ x, y, z :  ndarray coordinates
            ceval: backend to use: 
                - eval :  pure Numpy
                - numexpr.evaluate:  Numexpr """
        azimuth = ceval('arctan2(y,x)')
        # xy2 = ceval('x**2 + y**2')
        # elevation = ceval('arctan2(z, sqrt(xy2))')
        # r = eval('sqrt(xy2 + z**2)')
        return azimuth #, elevation, r

    def convertlist(self, longlist):
        tmp = list(chain.from_iterable(longlist))
        return np.array(tmp).reshape((len(longlist), len(longlist[0])))

    def callback_tau(self,cloud):
        if cloud: 
            print('cloud_callback')
            cloud_points = list(point_cloud2.read_points_list(cloud, field_names=("x", "y", "z"))) #, field_names=("x", "y", "z")
            
            points_arr = self.convertlist(cloud_points) #np.asarray(cloud_points)
            # print('cloud', points_arr)
            indices = np.logical_and(points_arr[:,0] > 0, points_arr[:,2]> -0.28)
            # print('maxiix', np.max(points_arr[:,2]))
            points_arr = points_arr[indices]
            indices = np.where(points_arr[:,2]<0.1)
            points_arr = points_arr[indices]
            azimuth = self.cart2sph(points_arr[:,0], points_arr[:,1], points_arr[:,2])
            np.cos(azimuth* (180/np.pi))
            points = np.column_stack((points_arr, azimuth))
            # print('points shpe', points)
            max = np.max(azimuth)
            min = np.min(azimuth)

            # print('size',np.size(points_arr_ ))
            ROI_el_ind = np.logical_and(points[:,3] > (max/3.5) , points[:,3] < (2*max/3))
            ROI_el = points[ROI_el_ind]
            if np.shape(ROI_el[:,0])[0] < 30:
                ROI_el_med = -1
            else:
                ROI_el_med = np.median(ROI_el[:,0])
            # print("el", np.shape(ROI_el[:,0]))

            ROI_l_ind = np.logical_and(points[:,3] > (max/10) , points[:,3] < (max/3.5))
            ROI_l = points[ROI_l_ind]
            if np.shape(ROI_l[:,0])[0] < 30:
                ROI_l_med = -1
            else:
                ROI_l_med = np.median(ROI_l[:,0])

            ROI_c_ind = np.logical_and(points[:,3] > (min/14) , points[:,3] < (max/14))
            ROI_c = points[ROI_c_ind]
            if np.shape(ROI_c[:,0])[0] < 30:
                ROI_c_med = -1
            else:
                ROI_c_med = np.median(ROI_c[:,0])

            ROI_r_ind = np.logical_and(points[:,3] > (min/3.5) , points[:,3] < (min/10))
            ROI_r = points[ROI_r_ind]
            if np.shape(ROI_r[:,0])[0] < 30:
                ROI_r_med = -1
            else:
                ROI_r_med = np.median(ROI_r[:,0])

            ROI_er_ind = np.logical_and(points[:,3] > (2*min/3) , points[:,3]< (min/3.5))
            ROI_er = points[ROI_er_ind]
            if np.shape(ROI_er[:,0])[0] < 30:
                ROI_er_med = -1
            else:
                ROI_er_med = np.median(ROI_er[:,0])
            # print("er", ROI_er_med)

            self.tau_val = [ROI_el_med, ROI_l_med, ROI_c_med, ROI_r_med, ROI_er_med]
            self.tau_val_bool = True
        else: 
            self.tau_val = None
            self.tau_val_bool = False 

    def save_tau_wksheet(self, tau_val):
        print('save_tau')
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
        print('image_call')
        if data:
            # start = time.time()
            # print("imagesfun")
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            # self.curr_image = im.fromarray(self.curr_image)
            self.curr_image_bool = True
            # print(f"Iteration_img:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
        else:
            self.curr_image = None
            self.curr_image_bool = False
            return

    def save_image(self, count : int, shared_path, curr_image): 
        print('image_saved')
        img_name= str(count) + '.png'
        path = shared_path + img_name
        curr_image = im.fromarray(curr_image)
        curr_image.save(path)

    def get_variables(self):
        print("get_variables")
        path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/"
        file = open(path + "variables.json")
        data = json.load(file)
        file.close()
        self.count = data["count"]
        self.count_2 = data["count_2"]
        
    def update_variables(self):
        print("update_variables")
        path = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/"
        file = open(path + "variables.json", "w")
        updated_data = {"count": self.count, "count_2": self.count_2}
        json.dump(updated_data, file)
        file.close()
    
    def collect_images(self):
        print('here_collect_img')
        if self.tau_val_bool and self.curr_image_bool:
            curr_image = self.curr_image
            print(curr_image)
            tau_val = self.tau_val
            print('img_bool',self.curr_image_bool)
            self.save_image(self.count, self.path_images,curr_image)
            print('tau_bool',self.tau_val_bool)
            self.save_tau_wksheet(tau_val)
            self.count += 1
            print(self.count)
            self.update_variables()

if __name__ == "__main__":
    rospy.init_node("collect_data")
    collect = collect_data()
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        collect.collect_images()
        r.sleep()   
    
