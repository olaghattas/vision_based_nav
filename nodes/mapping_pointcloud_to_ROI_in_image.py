#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2,PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge,  CvBridgeError 
import numpy as np
from tkinter import W
import rospy
from vision_based_navigation_ttt_ml.msg import TauComputation
import cv2
from sensor_msgs.msg import Image
import os
import pandas as pd
import xlsxwriter
from xlsxwriter import Workbook
import time
from itertools import chain

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
###########################################

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

#######################################################################################################################

class TauComp:
    def __init__(self):
        self.image_sub_name = "/realsense/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name,Image, self.img_callback)
        # Lidar Subscriber
        self.sub = rospy.Subscriber('/velodyne/points', PointCloud2, self.cloud_callback)
        # Tau Publisher
        self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=10)
        self.laser_pub_el = rospy.Publisher('/points_cloud_el', PointCloud2, queue_size=100)
        self.laser_pub_er = rospy.Publisher('/points_cloud_er', PointCloud2, queue_size=100)
        self.laser_pub_c = rospy.Publisher('/points_cloud_c', PointCloud2, queue_size=100)
        self.laser_pub_l = rospy.Publisher('/points_cloud_l', PointCloud2, queue_size=100)
        self.laser_pub_r = rospy.Publisher('/points_cloud_r', PointCloud2, queue_size=100)
        self.laser_pub_ = rospy.Publisher('/points_cloud_', PointCloud2, queue_size=100)
        self.curr_image = None
        self.bridge = CvBridge()
        
    def img_callback(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "mono8") #"passthrough"
            # cv2.imshow("",curr_image)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)
            return
        # Get time stamp

        self.secs = data.header.stamp.secs
        self.nsecs = data.header.stamp.nsecs
        self.width = data.width
        self.height = data.height

    def convertlist(self,longlist):
        tmp = list(chain.from_iterable(longlist))
        return np.array(tmp).reshape((len(longlist), len(longlist[0])))

    def longlist2array(self,longlist):
         wide = [[row[c] for row in longlist] for c in range(len(longlist[0]))]
         return np.array(wide).T
    
    def cloud_(self, cloud, points_arr_):
        # start = time.time()
        cloud_points = list(point_cloud2.read_points_list(cloud, skip_nans=True, field_names = ("x", "y", "z")))
        # print(cloud_points)
        # print(f"cloud_point_list:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
        # print('len',cloud_points)
        # print('0', cloud_points[0][:])
        
        # xy = np.random.rand(1000000, 2).tolist()
        # print("xy", len(xy[1]))
        # print('cl_pts', cloud_points)

        # start = time.time()
        # points_arr = np.asarray(cloud_points)
        # print(f"list_to_array numpy:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

        # start = time.time()
        points_arr= self.convertlist(cloud_points) #np.asarray(cloud_points)
        # print(f"list_to_array conv:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

        # start = time.time()
        # points_arr_long =  self.longlist2array(cloud_points)  #np.asarray(cloud_points)
        # print(f"list_to_array long:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

        # start = time.time()
        # points_arr = np.asarray(list(cloud_points))
        # print(f"list_to_array list:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

        # print("Equal_1", np.array_equal(points_arr_, points_arr_conv))
        # print("Equal_2", np.array_equal(points_arr_, points_arr_long))
        # print("Equal_3", np.array_equal(points_arr_conv, points_arr_long))
        
        # diffs = points_arr - points_arr_conv
        # indices = np.where(diffs>0)
        # diffs = diffs[indices]
        # with np.printoptions(threshold=np.inf):
        #     print('1', diffs)
        # print('2',points_arr_conv)
        # print('3',points_arr_long)
        # print('arra', points_arr)

        # print("indicesssss------")
        # print(points_arr[:,0] > 0)
        # print(points_arr[:,2] > -0.2)
        # start = time.time()
        indices = np.logical_and(points_arr[:,0] > 0, points_arr[:,2]> -0.28)
        points_arr_ = points_arr[indices]
        print('size',np.size(points_arr_ ))
        # print(f"loop:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

        # start = time.time()
        # for j in points_arr:
        #     # remove points behind the velodyne and on the floor
        #     if j[0]>0 and j[2]>-0.28: #j[0] x depth, j[1] y width, j[2] z height
        #         if points_arr_ is not None:
        #             points_arr_ = np.append(points_arr_, np.array([j]), axis=0)
        #         else:
        #             points_arr_ = np.array([j])
        # print(f"loop:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
        
        # print(np.size(points_arr_[:,0]))
        # print("Equal", np.array_equal(points_arr_, points_arr__))
        # print(points_arr_)
        return points_arr_
            
    def cloud_callback(self, data):
        start = time.time()
        if self.curr_image is not None:
            curr_image = self.curr_image
            points_arr_ = None
            points_arr_el = None
            points_arr_l = None
            points_arr_c = None
            points_arr_r = None
            points_arr_er = None
            
            if data is not None:
                # start = time.time()
                points_arr_ = self.cloud_(data, points_arr_)
                # print(f"cloud:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

                leng = len(points_arr_[:,0])

                ### In garden environment ###
                # points_arr_er = points_arr_[int(1.25*leng/4) : int(1.79*leng/4)]
                # points_arr_r = points_arr_[int(1.79*leng/4) : int(2.2*leng/4)]
                # points_arr_c = points_arr_[int(2.25*leng/4) : int(2.4*leng/4)]
                # points_arr_l = points_arr_[int(2.5*leng/4) : int(2.9*leng/4)]
                # points_arr_el = points_arr_[int(2.9*leng/4) : int(3.5*leng/4)]
                # start = time.time()
                end_er = int(1.65*leng/4)
                end_l = int(2.55*leng/4)
                points_arr_er = points_arr_[int(1.1*leng/4) : end_er]
                points_arr_r = points_arr_[end_er : int(2.05*leng/4)]
                points_arr_c = points_arr_[int(2.08*leng/4) : int(2.18*leng/4)]
                points_arr_l = points_arr_[int(2.2*leng/4) :end_l ]
                points_arr_el = points_arr_[end_l: int(3.2*leng/4)]
                # print(f"roi_points:\tTime taken: {(time.time()-start)*10**3:.03f}ms")
                

                # start = time.time()
                minimum = 10
                # print(np.size(points_arr_el[:,0]))
                if np.size(points_arr_el[:,0]) > minimum:
                    points_arr_el_median = np.median(points_arr_el[:,0])
                else:
                    points_arr_el_median = -1
                # print("med_el", points_arr_el_median)

                # print(np.size(points_arr_l[:,0]))
                if np.size(points_arr_l[:,0]) > minimum:
                    points_arr_l_median = np.median(points_arr_l[:,0])
                else:
                    points_arr_l_median = -1
                # print("med_l", points_arr_l_median)

                # print(np.size(points_arr_c[:,0]))
                if np.size(points_arr_c[:,0]) > minimum:
                    points_arr_c_median = np.median(points_arr_c[:,0])
                else:
                    points_arr_c_median = -1
                # print("med_c", points_arr_c_median)

                # print(np.size(points_arr_r[:,0]))
                if np.size(points_arr_r[:,0]) > minimum:
                    points_arr_r_median = np.median(points_arr_r[:,0])
                else:
                    points_arr_r_median = -1
                # print("med_r", points_arr_r_median)

                # print(np.size(points_arr_er[:,0]))
                if np.size(points_arr_er[:,0]) > minimum:
                    points_arr_er_median = np.median(points_arr_er[:,0])
                else:
                    points_arr_er_median = -1
                # print("med_er", points_arr_er_median)
                # print(f"medians:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

                # start = time.time()
                set_limit(self.width, self.height)
                   
                # Publish Tau values data to rostopic
                # Creation of TauValues.msg
                msg = TauComputation()
                msg.header.stamp.secs =  self.secs
                msg.header.stamp.nsecs =  self.nsecs
                msg.height = self.height
                msg.width = self.width

                msg.tau_el = points_arr_el_median
                msg.tau_er = points_arr_er_median
                msg.tau_l = points_arr_l_median
                msg.tau_r = points_arr_r_median
                msg.tau_c = points_arr_c_median
                self.tau_values.publish(msg)
                # print(f"msg:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

                # Draw the ROIs with their TTT values
                draw_image_segmentation(curr_image, points_arr_el_median , points_arr_er_median, points_arr_l_median, points_arr_r_median, points_arr_c_median)
                self.publish_cloud(data, points_arr_el,points_arr_l, points_arr_c, points_arr_r, points_arr_er,points_arr_)
        print(f"wholefunc:\tTime taken: {(time.time()-start)*10**3:.03f}ms")

    def publish_cloud(self, cloud, points_arr_el,points_arr_l, points_arr_c, points_arr_r, points_arr_er, points_arr_):
        # publish point cloud message
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
            ]

        header = Header()
        header.frame_id = cloud.header.frame_id
        header.stamp = cloud.header.stamp

        pc2_ = point_cloud2.create_cloud(header, fields,points_arr_)
        self.laser_pub_.publish(pc2_)

        pc2_el = point_cloud2.create_cloud(header, fields,points_arr_el)
        self.laser_pub_el.publish(pc2_el)

        pc2_l = point_cloud2.create_cloud(header, fields,points_arr_l)
        self.laser_pub_l.publish(pc2_l)

        pc2_c = point_cloud2.create_cloud(header, fields,points_arr_c)
        self.laser_pub_c.publish(pc2_c)

        pc2_r = point_cloud2.create_cloud(header, fields,points_arr_r)
        self.laser_pub_r.publish(pc2_r)

        pc2_er = point_cloud2.create_cloud(header, fields,points_arr_er)
        self.laser_pub_er.publish(pc2_er)


if __name__ == '__main__':
    rospy.init_node("tau_v", anonymous=False)
    tau = TauComp()   
    rospy.spin()   
    
      