#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2,PointField,LaserScan
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import numpy as np
from vision_based_navigation_ttt_ml.msg import TauComputation
import cv2
import os
import pandas as pd
from itertools import chain
from numpy import arctan2, sqrt
import numexpr as ne
from cv_bridge import CvBridgeError, CvBridge
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
def draw_image_segmentation(curr_image, final_tau_left_e, final_tau_right_e, final_tau_left, final_tau_right, final_tau_centre,\
     tau_el, tau_er, tau_l, tau_r, tau_c ):

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
    cv2.putText(color_image, str(round(final_tau_left_e, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/3)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(final_tau_right_e, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 3)),
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
    cv2.putText(color_image, str(round(final_tau_left, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 3)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(final_tau_right, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 3)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
  
    # Centre 
    cv2.rectangle(color_image, (x_init_c, y_init_c), (x_end_c, y_end_c), color_red, linewidth)
    cv2.putText(color_image, str(round(tau_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 2)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(final_tau_centre, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 3)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('ROIs Representation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROIs Representation', (600, 600))
    cv2.imshow('ROIs Representation', color_image)
    cv2.waitKey(10)

class Velodyne:
    def __init__(self):
        # Velodyne Subscriber
        self.sub = rospy.Subscriber('/velodyne/points', PointCloud2, self.cloud_callback)
        self.vel_cloud_pub = rospy.Publisher('/points_cloud_', PointCloud2, queue_size=1)
        self.vel_el_pub = rospy.Publisher('/points_el_', PointCloud2, queue_size=1)
        self.vel_er_pub = rospy.Publisher('/points_er_', PointCloud2, queue_size=1)
        self.vel_l_pub = rospy.Publisher('/points_l_', PointCloud2, queue_size=1)
        self.vel_r_pub = rospy.Publisher('/points_r_', PointCloud2, queue_size=1)
        self.vel_c_pub = rospy.Publisher('/points_c_', PointCloud2, queue_size=1)
        # Lidar Subscriber
        self.sub_lidar = rospy.Subscriber('/front/scan', LaserScan, self.callback)
        # Camera Subscriber
        self.image_sub_name = "/realsense/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, callback=self.callback_img,queue_size=1,buff_size=2**18) #the buff_size=2**18 avoids delays due to the queue buffer being too small for images
        
        # self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        self.bridge = CvBridge()
        self.tau_el = -1
        self.tau_er = -1
        self.tau_l = -1
        self.tau_r = -1
        self.tau_c = -1
        self.ranges = None
        self.linear_x_vel = 1
        self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=1)

    def convertlist(self,longlist):
        tmp = list(chain.from_iterable(longlist))
        return np.array(tmp).reshape((len(longlist), len(longlist[0])))

    def callback_img(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)
            return
        self.secs = data.header.stamp.secs
        self.nsecs = data.header.stamp.nsecs
        self.width = data.width
        self.height = data.height

    def callback(self, msg):
        print("here")
        start_ind  = 230 #0
        end_ind = 488 #len(msg.ranges) - 1   #488 #
        # print('ol',msg.angle_max)
        self.angle_min = msg.angle_min + start_ind * msg.angle_increment
        self.angle_max = msg.angle_min + end_ind * msg.angle_increment
        self.increments = msg.angle_increment
        self.ranges = msg.ranges[230:489]    

    def cart2sph(self, x,y,z, ceval=ne.evaluate):
        """ x, y, z :  ndarray coordinates
            ceval: backend to use: 
                - eval :  pure Numpy
                - numexpr.evaluate:  Numexpr """
        azimuth = ceval('arctan2(y,x)')
        # xy2 = ceval('x**2 + y**2')
        # elevation = ceval('arctan2(z, sqrt(xy2))')
        # r = eval('sqrt(xy2 + z**2)')
        return azimuth #, elevation, r
    
    def cloud_callback(self, cloud):
        # start_time = time.time()
        curr_image = self.curr_image  
        # rate = rospy.Rate(1) #hz
        cloud_points = list(point_cloud2.read_points_list(cloud, field_names=("x", "y", "z"))) #, field_names=("x", "y", "z")
        # print('cloud', cloud_points)
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
        if np.shape(ROI_el[:,0]) < 30:
            ROI_el_med = -1
        else:
            ROI_el_med = np.median(ROI_el[:,0])
        # print("el", np.shape(ROI_el[:,0]))

        ROI_l_ind = np.logical_and(points[:,3] > (max/10) , points[:,3] < (max/3.5))
        ROI_l = points[ROI_l_ind]
        if np.shape(ROI_l[:,0]) < 30:
            ROI_l_med = -1
        else:
            ROI_l_med = np.median(ROI_l[:,0])

        ROI_c_ind = np.logical_and(points[:,3] > (min/14) , points[:,3] < (max/14))
        ROI_c = points[ROI_c_ind]
        if np.shape(ROI_c[:,0]) < 30:
            ROI_c_med = -1
        else:
            ROI_c_med = np.median(ROI_c[:,0])

        ROI_r_ind = np.logical_and(points[:,3] > (min/3.5) , points[:,3] < (min/10))
        ROI_r = points[ROI_r_ind]
        if np.shape(ROI_r[:,0]) < 30:
            ROI_r_med = -1
        else:
            ROI_r_med = np.median(ROI_r[:,0])

        ROI_er_ind = np.logical_and(points[:,3] > (2*min/3) , points[:,3]< (min/3.5))
        ROI_er = points[ROI_er_ind]
        if np.shape(ROI_er[:,0]) < 30:
            ROI_er_med = -1
        else:
            ROI_er_med = np.median(ROI_er[:,0])
            
        # print('Duration: {}'.format(time.time() - start_time))
        # print('ring',points_arr_rings[:,0:3])
        # self.publish_cloud(cloud, points_arr_azimuth[:,:3])
############################################################
        # publish point cloud message
        # fields = [PointField('x', 0, PointField.FLOAT32, 1),
        #     PointField('y', 4, PointField.FLOAT32, 1),
        #     PointField('z', 8, PointField.FLOAT32, 1)
        #     ]

        # header = Header()
        # header.frame_id = cloud.header.frame_id
        # header.stamp = cloud.header.stamp

        # pc2_cloud = point_cloud2.create_cloud(header, fields, cloud)
        # self.vel_cloud_pub.publish(pc2_cloud)

        # pc2_el = point_cloud2.create_cloud(header, fields, ROI_el[:,:3])
        # self.vel_el_pub.publish(pc2_el)

        # pc2_er = point_cloud2.create_cloud(header, fields, ROI_er[:,:3])
        # self.vel_er_pub.publish(pc2_er)

        # pc2_l = point_cloud2.create_cloud(header, fields, ROI_l[:,:3])
        # self.vel_l_pub.publish(pc2_l)

        # pc2_r = point_cloud2.create_cloud(header, fields, ROI_r[:,:3])
        # self.vel_r_pub.publish(pc2_r)

        # pc2_c = point_cloud2.create_cloud(header, fields, ROI_c[:,:3])
        # self.vel_c_pub.publish(pc2_c)

        # Publish Tau values data to rostopic
        # Creation of TauValues.msg
        msg = TauComputation()
        msg.header.stamp.secs =  self.secs
        msg.header.stamp.nsecs =  self.nsecs
        msg.height = self.height
        msg.width = self.width

        msg.tau_el = ROI_el_med
        msg.tau_er = ROI_er_med
        msg.tau_l = ROI_l_med
        msg.tau_r = ROI_r_med
        msg.tau_c = ROI_c_med
        self.tau_values.publish(msg)

        self.get_tau_values()

        draw_image_segmentation(curr_image,ROI_el_med,ROI_er_med,ROI_l_med,ROI_r_med,ROI_c_med, self.tau_el, self.tau_er, self.tau_l, self.tau_r, self.tau_c)

    def get_tau_values(self):
        start = time.time()
        set_limit(self.width, self.height)
        if self.curr_image is not None:
            curr_image = self.curr_image  
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
                self.tau_er = tau_er
                self.tau_l = tau_l
                self.tau_r = tau_r
                self.tau_c= tau_c

    def publish_cloud(self, cloud, points_arr_rings):
        # publish point cloud message
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
            ]

        header = Header()
        header.frame_id = cloud.header.frame_id
        header.stamp = cloud.header.stamp

        # pc2_cloud = point_cloud2.create_cloud(header, fields, cloud)
        # self.vel_cloud_pub.publish(pc2_cloud)

        pc2_rings = point_cloud2.create_cloud(header, fields, points_arr_rings)
        self.vel_rings_pub.publish(pc2_rings)

if __name__ == '__main__':
    rospy.init_node("tau_v", anonymous=False)
    vel = Velodyne()
    rospy.spin()   
    
      
