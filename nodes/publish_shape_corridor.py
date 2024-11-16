#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16MultiArray
import tensorflow as tf
import os
from cv_bridge import CvBridgeError, CvBridge
from sensor_msgs.msg import Image
import numpy as np
import cv2

class Predict_Shape():
    def __init__(self):
        image_sub_name = "/realsense/color/image_raw"
        image_sub = rospy.Subscriber(image_sub_name, Image, self.callback_img)
        self.bridge = CvBridge()
        
        path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/trained_model_parameters/"
        self.model = tf.keras.models.load_model(path + "determine_shape_model_res50.h5", compile=False) #, custom_objects=ak.CUSTOM_OBJECTS
        self.model.compile(optimizer = 'adam', loss = 'mae', metrics = ['MeanSquaredError', 'mean_absolute_error']) #Paste it here
        self.curr_image = None
        self.pub = rospy.Publisher('chatter', Int16MultiArray, queue_size=10) 
    
    def talker(self,pred):
       
        direction = Int16MultiArray(data = pred) #(left,right)
        rospy.loginfo(direction)
        self.pub.publish(direction)
        rate.sleep()

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

    def cnn_prediction(self):
        if self.curr_image is not None:
            img_size = 250
            curr_image = self.curr_image
            img = cv2.resize(curr_image,(img_size,img_size))
            img = tf.expand_dims(img, 0)
            img = np.asarray(img)
            
            shape_pred = self.model.predict({"input_1": img})
        
            self.talker([round(shape_pred[0][0]),round(shape_pred[0][1]),round(shape_pred[0][2])])

if __name__ == '__main__':
    shape = Predict_Shape()
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(5) # 5hz
    while not rospy.is_shutdown():
        shape.cnn_prediction()
        rate.sleep()