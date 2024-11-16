#!/usr/bin/env python3
import rospy 
from gazebo_msgs.msg import ModelStates
import csv
import os
import json
from cv_bridge import CvBridgeError, CvBridge
from vision_based_navigation_ttt_ml.msg import TauComputation
from sensor_msgs.msg import Image
from PIL import Image as im
import xlsxwriter
from tf.transformations import euler_from_quaternion

class Save_Data():
    def __init__(self):
        self.tau_values= rospy.Subscriber("/tau_values", TauComputation, self.callback_tau)
        self.image_sub_name = "/realsense/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        self.bridge = CvBridge()
        self.get_variables()
        self.count_2 += 1
        self.update_variables()
        self.curr_image = None
        self.path_folder = os.environ["HOME"] + "/catkin_ws/src/vision_based_navigation_ttt_ml/"
        self.path_tau = self.path_folder + "shape_label_tau/"
        vel = '1' # velocity 
        self.folder_name = 'shape_label_train/training_images_' + str(self.count_2) + '_v_' + vel + '/'
        self.path_images = self.path_folder + self.folder_name
        os.mkdir(self.path_folder + self.folder_name)
        self.tau_val = None
        self.csv_file_name = 'labels_file_tr.csv'
        if os.stat(self.path_folder + self.csv_file_name).st_size == 0:
            self.dict_labels = {}
        else:
            self.dict_labels = self.read_from_csv(self.path_folder + self.csv_file_name)

        
    def callback_img(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.curr_image = im.fromarray(self.curr_image)
        except CvBridgeError as e:
            print(e)
            return
    
    def save_dict_to_csv(self, dictionary):
        # Open the file in writing mode (no blank lines)
        with open(self.path_folder + self.csv_file_name, 'w', newline='') as f:
            # Create a CSV writer object
            writer = csv.writer(f)
            # Write one key-value tuple per row
            for row in dictionary.items():
                writer.writerow(row)

    def read_from_csv(self, csv_filename):
        dict_from_csv = {}
        with open(csv_filename, mode='r') as inp:
            reader = csv.reader(inp)
            dict_from_csv = {rows[0]:rows[1] for rows in reader}
        return dict_from_csv

    def callback_tau(self,data):
        if data:
            self.tau_val = [data.tau_el, data.tau_l, data.tau_c, data.tau_r, data.tau_er]

    def data_collection(self, label): # label (left, right, straight)
        # print('curr',self.curr_image, 'tau_va', self.tau_val)
        print('data_coll')
        if self.curr_image is not None and self.tau_val is not None:
            self.dict_labels[self.count] = label
            # print('img_num',self.images[self.current_image].split('.')[0].split('/')[8])
            # print('dict',self.dict_labels)
            curr_image = self.curr_image
            print('*********************',label)
            self.save_dict_to_csv(self.dict_labels)
            try: 
            # print(np.type(curr_image))
                self.save_image(self.count, self.path_images, curr_image)
            except:
                print('error saing image')
                return
            
            wb_tau =  xlsxwriter.Workbook(self.path_tau + "tau_value" + str(self.count) + ".xlsx")
            wksheet_tau = wb_tau.add_worksheet()
            curr_image = self.curr_image
            tau_val = self.tau_val

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

            self.count += 1
            print(self.count)
            self.update_variables()
    
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

class Env(Save_Data):
    def __init__(self):
        self.sub_robot_state = rospy.Subscriber("/gazebo/model_states",ModelStates,self.callback_state)
        self.theta = None
        super().__init__()
        # Save_Data(self)
        self.label = None

    def callback_state(self,data):
        for i in range(len(data.name)):
            if data.name[i] == "jackal":
                self.x = data.pose[i].position.x
                self.y = data.pose[i].position.y
                rot_q = data.pose[i].orientation
                roll,pitch,theta = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
                self.theta = theta
                # print('angles',theta)
        
    def L_shaped_env(self):
        if self.theta is not None:
            print('theta',self.theta)
            print('x',self.x,'y',self.y)

            if 3 <= self.x <= 10.9 and -2.4 <= self.y <= -0.8: #if robot is in first_corridor
                print('1')
                if -3.15 <= self.theta <= -2  or 2 <= self.theta <= 3.15:
                    print('1.1')
                    self.label = (0,1,0) # rigth turn
                elif -0.7 <= self.theta <= 0.7:
                    print('1.2')
                    self.label = (0,0,0) # no turns
    
            elif 0.4 <= self.x <= 2 and 0.9 <= self.y <= 10: #if robot is in second_corridor
                print('2')
                if 0.6 <= self.theta <= 2.2:
                    print('2.1')
                    self.label = (0,0,0) #no turns
                elif  -2.4 <= self.theta <= -0.6:
                    print('2.2')
                    self.label = (1,0,0) # left
            
            else:
                print('3')
                self.label = (0,0,0) # no turns
            
            print('label', self.label)
            if self.label is not None:
                self.data_collection(self.label)

    def complex_env(self):
        if self.theta is not None:
            print('theta',self.theta)
            print('x',self.x,'y',self.y)

            if 3 <= self.x <= 10.9 and -2.4 <= self.y <= -0.8: #if robot is in first corridor
                print('1')
                if -3.15 <= self.theta <= -2  or 2 <= self.theta <= 3.15:
                    print('1.1')
                    self.label = (0,1,0) # rigth turn
                elif -0.7 <= self.theta <= 0.7:
                    print('1.2')
                    self.label = (0,0,0) # no turns
    
            elif 0.4 <= self.x <= 2 and -1.2 <= self.y <= 10: #if robot is in second corridor
                print('2')
                if 0.6 <= self.theta <= 2.2:
                    print('2.1')
                    self.label = (1,0,0) # left 
                elif  -2.4 <= self.theta <= -0.6:
                    print('2.2')
                    self.label = (1,0,0) # left
            
            elif -4.6 <= self.x <= -0.43 and 7.9 <= self.y <= 10: #if robot is in third corridor
                if -3.15 <= self.theta <= -2  or 2 <= self.theta <= 3.15:
                    print('1.1')
                    self.label = (1,0,1) # intersection straight and left
                elif -0.7 <= self.theta <= 0.7:
                    print('1.2')
                    self.label = (0,1,0) # right turn

            elif -15 <= self.x <= -6 and 7.9 <= self.y <= 10: #if robot is in fourth corridor
                if -3.15 <= self.theta <= -2  or 2 <= self.theta <= 3.15:
                    print('1.1')
                    self.label = (0,0,0) # no turns 
                elif -0.7 <= self.theta <= 0.7:
                    print('1.2')
                    self.label = (0,1,1) # intersection straight and right

            elif -6.3 <= self.x <= -4.7 and -0.43 <= self.y <= 10: #if robot is in fifth corridor
                print('2')
                if 0.6 <= self.theta <= 2.2:
                    print('2.1')
                    self.label = (1,1,0) # left and right turns
                elif  -2.4 <= self.theta <= -0.6:
                    print('2.2')
                    self.label = (0,0,0) # no turns
            else:
                print('3')
                self.label = None # no turns

            # print('label', self.label)
            # if self.label is not None:
            #     self.data_collection(self.label)

    def oval_env(self):
        if self.theta is not None:
            print('theta',self.theta)
            print('x',self.x,'y',self.y)

            if -7.8 <= self.x <= 7.8 and -2.4 <= self.y <= -0.6: #if robot is in first corridor
                print('1')
                if -3.15 <= self.theta <= -2  or 2 <= self.theta <= 3.15:
                    print('1.1')
                    self.label = (0,1,0) #  right turn
                    print('right')
                elif -0.7 <= self.theta <= 0.7:
                    print('1.2')
                    self.label = (1,0,0) # left turn
                    print('left')
        
            elif -9.5 <= self.x <= -7.6 and -1.5 <= self.y <= 10.5: #if robot is in second corridor
                print('2')
                if 0.6 <= self.theta <= 2.2:
                    print('2.1')
                    self.label = (0,1,0) # right
                    print('right')
                elif  -2.4 <= self.theta <= -0.6:
                    print('2.2')
                    self.label = (1,0,0) # left
                    print('left')

            elif -7.8 <= self.x <= 7.8 and 9 <= self.y <= 11: #if robot is in third corridor
                print('1')
                if -3.15 <= self.theta <= -2  or 2 <= self.theta <= 3.15:
                    print('1.1')
                    self.label = (1,0,0) # left turn
                    print('left')
                elif -0.7 <= self.theta <= 1.7:
                    print('1.2')
                    self.label = (0,1,0) #  right turn
                    print('right')
                    
        
            elif 7.3 <= self.x <= 9.3 and -1.5 <= self.y <= 10.5: #if robot is in fourth corridor
                print('2')
                if 0.6 <= self.theta <= 2.2:
                    print('2.1')
                    
                    self.label = (1,0,0) # left
                    print('left')
                elif  -2.4 <= self.theta <= -0.6:
                    print('2.2')
                    self.label = (0,1,0) # right
                    print('right')
            else:
                print('3')
                self.label = None

            print('label', self.label)
            if self.label is not None:
                self.data_collection(self.label)
                
    def garden_house_env(self):
        pass

    def T_shaped_env(self):
        pass
        
if __name__ == "__main__":
    rospy.init_node("label")  
    collect = Env()
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): 
        collect.oval_env()
        r.sleep()  
    