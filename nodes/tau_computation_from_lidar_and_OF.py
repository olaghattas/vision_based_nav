#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from vision_based_navigation_ttt_ml.msg import OpticalFlow
from vision_based_navigation_ttt_ml.msg import TauComputation
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
import cv2
from sensor_msgs.msg import LaserScan
import os
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt

# Extreme left and extreme right
x_init_el = 0
y_init_el = 0
x_end_el = 0
y_end_el = 0

x_init_er = 0
y_init_er = 0
x_end_er = 0
y_end_er = 0

# Left and right
x_init_l = 0
y_init_l = 0
x_end_l = 0
y_end_l = 0

x_init_r = 0
y_init_r = 0
x_end_r = 0
y_end_r = 0

# Centre
x_init_c = 0
y_init_c = 0
x_end_c = 0
y_end_c = 0

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

# Filtering procedure for the TTT values
def tau_filtering_mad(vector):
    range = 1
    sorted_indices = np.argsort(vector)
    sorted_vector = vector[sorted_indices]
    median_sorted = np.median(sorted_vector)
    # # method using median absolute deviation
    mad = np.median(np.absolute(sorted_vector - median_sorted))
    sorted_indices_2 = sorted_indices[(sorted_vector < (median_sorted + range*mad)) & (sorted_vector > (median_sorted - filter*mad))]
    vector = sorted_vector[sorted_indices_2]
    return vector

def tau_filtering(vector):
    perc_TTT_val_discarded = 0.15
    jump = int(perc_TTT_val_discarded * np.size(vector))
    vector = np.sort(vector)
    # plt.scatter(vector,np.zeros(len(vector)))
    # plt.title("before")
    # plt.show()
    vector = np.delete(vector, range(jump))
    vector = np.delete(vector, range(np.size(vector) - jump, np.size(vector)))
    # plt.scatter(vector,np.zeros(len(vector)))
    # plt.title("after")
    # plt.show()
    return vector

# Computation of the average TTT
def tau_final_value(self, vector, cnt):

    if cnt >= self.min_TTT_number:
        mean = np.sum(vector) / cnt
    else:
        mean = -1

    return mean

# Visual representation of the ROIs with the average TTT values
def draw_image_segmentation(curr_image, final_tau_left_e, final_tau_right_e, final_tau_left, final_tau_right, final_tau_centre,\
     tau_el, tau_er, tau_l, tau_r, tau_c, error_er,error_el,error_r,error_l,error_c ):

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
    cv2.putText(color_image, str(round(error_el, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/1.5)),
                font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(error_er, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 1.5)),
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
    cv2.putText(color_image, str(round(error_l, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 1.5)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(error_r, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 1.5)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Centre 
    cv2.rectangle(color_image, (x_init_c, y_init_c), (x_end_c, y_end_c), color_red, linewidth)
    cv2.putText(color_image, str(round(tau_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 2)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(final_tau_centre, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 3)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(error_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 1.5)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('ROIs Representation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROIs Representation', (600, 600))
    cv2.imshow('ROIs Representation', color_image)
    cv2.waitKey(10)

class TauComputationClass:
    def __init__(self):

        ######## IMPORTANT PARAMETERS: ########
        # Minimum number of features needed to compute the average TTT for each ROI
        self.min_TTT_number = 10
        self.image_sub_name = "/realsense/color/image_raw"
        #######################################

        # First time that the callback is called
        self.first_time = True
        # Initialize current image
        self.curr_image = None

        # Initialize Image acquisition
        self.bridge = CvBridge()
        # OpticalFlowData Subscriber
        self.of_sub = rospy.Subscriber("optical_flow", OpticalFlow, self.callback_of)
        # Raw Image Subscriber
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        # Tau Computation message Publisher
        self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=10)
        # Lidar Subscriber
        self.sub = rospy.Subscriber('/front/scan', LaserScan, self.callback)
        self.ranges = None
        self.increments = None
        self.linear_x_vel = 1
        self.angle_min = None
        self.angle_max = None
        self.relative_error_er = 0
        self.relative_error_el = 0
        self.relative_error_c = 0
        self.relative_error_r = 0
        self.relative_error_l = 0
        self.er_error_array = []
        self.r_error_array = []
        self.c_error_array = []
        self.l_error_array = []
        self.el_error_array = []
        
    def callback(self, msg):
            start_ind  = 230 #0
            end_ind = 488 #len(msg.ranges) - 1   #488 #
            # print('ol',msg.angle_max)
            self.angle_min = msg.angle_min + start_ind * msg.angle_increment
            self.angle_max = msg.angle_min + end_ind * msg.angle_increment
            self.increments = msg.angle_increment
            self.ranges = msg.ranges[230:489]
            # print(self.ranges)

    # Callback for the Optical flow topic
    def callback_of(self, data):

        img_width = data.width
        img_height = data.height

        # Coordinates at the center of the image
        xc = np.floor(img_width/2)
        yc = np.floor(img_height/2)
        # Express all points coordinate with respect to center of the image
        x = data.x - xc
        y = data.y - yc

        # Definition of the five ROIs only the first time the callback is called
        if self.first_time:
            set_limit(img_width, img_height)
            self.first_time = False

        # Initialization tau computation extreme left and extreme right
        tau_right_e = np.array([])
        tau_left_e = np.array([])
        count_left_e = 0
        count_right_e = 0

        # Initialization tau computation left and right
        tau_right = np.array([])
        tau_left = np.array([])
        count_left = 0
        count_right = 0

        # Initialization tau computation centre
        tau_centre = np.array([])
        count_centre = 0

        # TTT values computation
        for i in range(len(x)):
            print('here')
            # Extreme left and right
            if (x[i] >= (x_init_er - xc)) and (y[i] >= (y_init_er - yc)) and (y[i] <= (y_end_er - yc)):
                tau_right_e = np.append(tau_right_e, (x[i]**2 + y[i]**2)**0.5 / (data.vx[i]**2 + data.vy[i]**2)**0.5)
                count_right_e += 1
            if (x[i] <= (x_end_el - xc)) and (y[i] >= (y_init_el - yc)) and (y[i] <= (y_end_el - yc)):
                tau_left_e = np.append(tau_left_e, (x[i]**2 + y[i]**2)**0.5 / (data.vx[i]**2 + data.vy[i]**2)**0.5)
                count_left_e += 1

            # Left and right
            if (x[i] >= (x_init_r - xc)) and (x[i] <= (x_end_r - xc)) \
                    and (y[i] >= (y_init_r - yc)) and (y[i] <= (y_end_r - yc)):
                tau_right = np.append(tau_right, (x[i]**2 + y[i]**2)**0.5 / (data.vx[i]**2 + data.vy[i]**2)**0.5)
                count_right += 1
            if (x[i] <= (x_end_l - xc)) and (x[i] >= (x_init_l - xc)) \
                    and (y[i] >= (y_init_l - yc)) and (y[i] <= (y_end_l - yc)):
                tau_left = np.append(tau_left, (x[i]**2 + y[i]**2)**0.5 / (data.vx[i]**2 + data.vy[i]**2)**0.5)
                count_left += 1

            # Centre
            if (x[i] >= (x_init_c - xc)) and (x[i] <= (x_end_c - xc)) \
                    and (y[i] >= (y_init_c - yc)) and (y[i] <= (y_end_c - yc)):
                tau_centre = np.append(tau_centre,
                                     (x[i] ** 2 + y[i] ** 2) ** 0.5 / (data.vx[i] ** 2 + data.vy[i] ** 2) ** 0.5)
                count_centre += 1

        # Filtering TTT values for each ROI
        # Extreme right and left
        tau_right_e = tau_filtering_mad(tau_right_e)
        tau_left_e = tau_filtering_mad(tau_left_e)
        # Right and left
        tau_right = tau_filtering_mad(tau_right)
        tau_left = tau_filtering_mad(tau_left)
        # Centre
        tau_centre = tau_filtering_mad(tau_centre)
        # Extreme right and left
        final_tau_left_e = tau_final_value(self, tau_left_e, count_left_e)
        final_tau_right_e = tau_final_value(self, tau_right_e, count_right_e)
        # print("Tau right Extreme: " + str(final_tau_right_e))
        # print("Tau left Extreme: " + str(final_tau_left_e))
        # Right and left
        final_tau_left = tau_final_value(self, tau_left, count_left)
        final_tau_right = tau_final_value(self, tau_right, count_right)
        # print("Tau right: " + str(final_tau_right))
        # print("Tau left: " + str(final_tau_left))
        # Centre
        final_tau_centre = tau_final_value(self, tau_centre, count_centre)
        # print("Tau centre: " + str(final_tau_centre))

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

            # Publish Tau values data to rostopic
            # Creation of TauValues.msg
            msg = TauComputation()
            msg.header.stamp.secs = data.header.stamp.secs
            msg.header.stamp.nsecs = data.header.stamp.nsecs

            msg.height = data.height
            msg.width = data.width

            msg.tau_el = tau_el
            msg.tau_er = tau_er
            msg.tau_l = tau_l
            msg.tau_r = tau_r
            msg.tau_c = tau_c
            self.tau_values.publish(msg)

        else:
            tau_el = -1
            tau_er = -1
            tau_l = -1
            tau_r = -1
            tau_c = -1
        if tau_er == np.inf:
            tau_er =-1
        if tau_el == np.inf:
            tau_el =-1
        if tau_r == np.inf:
            tau_r =-1
        if tau_l == np.inf:
            tau_l =-1
        if tau_c == np.inf:
            tau_c =-1

        if final_tau_right_e == np.inf:
            final_tau_right_e =-1
        if final_tau_left_e == np.inf:
            final_tau_left_e =-1
        if final_tau_right == np.inf:
            final_tau_right =-1
        if final_tau_left == np.inf:
            final_tau_left =-1
        if final_tau_centre == np.inf:
            final_tau_centre =-1
            
        self.relative_error_er = (tau_er - final_tau_right_e )
        self.relative_error_r = (tau_r - final_tau_right)
        self.relative_error_c = (tau_c - final_tau_centre)
        self.relative_error_l = (tau_l - final_tau_left)
        self.relative_error_el = (tau_el - final_tau_left_e)
       
        self.er_error_array.append(self.relative_error_er)
        self.el_error_array.append(self.relative_error_el)
        self.r_error_array.append(self.relative_error_r)
        self.l_error_array.append(self.relative_error_l)
        self.c_error_array.append(self.relative_error_c)
        
        error_er = (tau_er - final_tau_right_e) / tau_er
        error_el = (tau_el - final_tau_left_e) / tau_el
        error_r = (tau_r - final_tau_right) / tau_r
        error_l = (tau_l - final_tau_left) / tau_l
        error_c = (tau_c - final_tau_centre) / tau_c
        # Draw the ROIs with their TTT values
        draw_image_segmentation(self.curr_image, final_tau_left_e, final_tau_right_e, final_tau_left, final_tau_right, \
            final_tau_centre,tau_el, tau_er, tau_l, tau_r, tau_c, error_er, error_el, error_r, error_l, error_c)

    # Callback for the image topic
    def callback_img(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)
            return

    def on_rospy_shutdown(self):
            rospy.logwarn("Stopping")
            rospy.Rate(1).sleep()
            count = 1
            filter = 'mad_range_1_'

            path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/mad_ranges/"
            folder_name = filter + str(count) + '/'
            os.mkdir(path_folder + folder_name) 
            
            # Region_of_interest = [self.er_error_array, self.r_error_array, self.c_error_array,
            #          self.l_error_array,self.el_error_array]

            er_error_array = pd.DataFrame(self.er_error_array)
            # filepath_er = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/csv/er_error_array.xlsx"
            er_error_array.to_excel( path_folder + folder_name + 'er_error_array_' + filter + '.xlsx')

            el_error_array = pd.DataFrame(self.el_error_array)
            # filepath_el = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/csv/el_error_array.xlsx"
            el_error_array.to_excel(path_folder + folder_name + 'el_error_array_' + filter + '.xlsx')

            r_error_array = pd.DataFrame(self.r_error_array)
            # filepath_r = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/csv/r_error_array.xlsx"
            r_error_array.to_excel(path_folder + folder_name + 'r_error_array_' + filter + '.xlsx')

            l_error_array = pd.DataFrame(self.l_error_array)
            # filepath_l = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/csv/l_error_array.xlsx"
            l_error_array.to_excel(path_folder + folder_name + 'l_error_array_' + filter + '.xlsx')

            c_error_array = pd.DataFrame(self.c_error_array)
            # filepath_c = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/csv/c_error_array.xlsx"
            c_error_array.to_excel(path_folder + folder_name + 'c_error_array_' + filter + '.xlsx')

            # print(len(self.c_error_array))
            # fig=plt.figure(figsize=(10,7))
            # columns = 3
            # rows = 2
            # array = [self.er_error_array, self.el_error_array, self.r_error_array, self.l_error_array, self.c_error_array]
            # array_str = ['er_error_array','el_error_array','r_error_array','l_error_array','c_error_array']
            
            # for i in range(1, 6):
            #     fig.add_subplot(rows, columns, i)
            #     plt.plot(np.cumsum(array[i-1]),np.arange(0,np.size(array[i-1],0),1))### what you want you can plot
            #     plt.xlabel(array_str[i-1]) 
            #     plt.ylabel('steps') 
            #     plt.title(filter + str(count))
            # plt.show()

            # plt.figure(1, figsize=(15, 5))

            # plt.subplot(211)
            # plt.ylim(-5,5)
            # plt.scatter(np.arange(0,np.size(self.er_error_array,0),count),self.er_error_array, color='blue', marker='o', label='data')

            # plt.subplot(212)
            # # plt.xlim(-15,15)
            # plt.hist(self.er_error_array, color = 'blue', edgecolor = 'black',
            #         bins = int(180/5))

            # # plt.savefig(path + 'er_error_images/' + filter + str(count) +".png")


            # plt.figure(2, figsize=(15, 5))

            # plt.subplot(211)
            # plt.ylim(-5,5)
            # plt.scatter( np.arange(0,np.size(self.el_error_array,0),1),self.el_error_array, color='blue', marker='o', label='data')

            # plt.subplot(212)
            # # plt.xlim(-15,15)
            # plt.hist(self.el_error_array, color = 'blue', edgecolor = 'black',
            #         bins = int(180/5))

            # # plt.savefig(path + 'el_error_images/' + filter + str(count) +".png")

            # plt.figure(3, figsize=(15, 5))

            # plt.subplot(211)
            # plt.ylim(-5,5)
            # plt.scatter( np.arange(0,np.size(self.r_error_array,0),1),self.r_error_array, color='blue', marker='o', label='data')

            # plt.subplot(212)
            # # plt.xlim(-15,15)
            # plt.hist(self.r_error_array, color = 'blue', edgecolor = 'black',
            #         bins = int(180/5))

            # plt.savefig(path + 'r_error_images/' + filter + str(count) +".png")

            # plt.figure(4, figsize=(15, 5))

            # plt.subplot(211)
            # plt.ylim(-5,5)
            # plt.scatter( np.arange(0,np.size(self.l_error_array,0),1), self.l_error_array,color='blue', marker='o', label='data')

            # plt.subplot(212)
            # # plt.xlim(-15,15)
            # plt.hist(self.l_error_array, color = 'blue', edgecolor = 'black',
            #         bins = int(180/5))

            # plt.savefig(path + 'l_error_images/' + filter + str(count) +".png")

            # plt.figure(5, figsize=(15, 5))

            # plt.subplot(211)
            # plt.ylim(-5,5)
            # plt.scatter(np.arange(0,np.size(self.c_error_array,0),1),self.c_error_array, color='blue', marker='o', label='data')
            # plt.subplot(212)
            # # plt.xlim(-15,15)
            # plt.hist(self.c_error_array, color = 'blue', edgecolor = 'black',
            #         bins = int(180/5))
            # plt.savefig(path + 'c_error_images/' + filter + str(count) +".png")

def tau_computation():
    rospy.init_node("tau_computation", anonymous=False)
    tr = TauComputationClass()
    rospy.on_shutdown(tr.on_rospy_shutdown)
    rospy.spin()


if __name__ == '__main__':
    tau_computation()
    
