#!/usr/bin/env python3

#  USED TO CHANGE THE COLOR OF THE BOXES IN THE ENVIIRONMENT
#  NEEDS TO BE RUN BEFORE lAUNCHIN THE GAZEBO ENVIRONMENT OR ADD IT TO THE LAUNCH FILE
import random
from PIL import Image  
import PIL  
import os
import rospy

def change_envri():
    num_r_img = 6
    num_b_img = 6
    num_g_img = 6

    path_red = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/images_with_colored_bkg/red/red_"
    path_green =  os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/images_with_colored_bkg/green/green_"
    path_blue =   os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/images_with_colored_bkg/blue/blue_"

    num_r_1 = random.randint(0, num_r_img)
    num_g_1 = random.randint(0, num_b_img)
    num_b_1 = random.randint(0, num_g_img)
    print(num_r_1, num_g_1, num_b_1)
    red_img_1 = Image.open(path_red + str(num_r_1) + '.png')  
    green_img_1 = Image.open(path_green + str(num_g_1) + '.png')  
    blue_img_1 = Image.open(path_blue + str(num_b_1) + '.png')  
    path_1s = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/models/1_stack/materials/textures/"
    red_img_1.save(path_1s + 'red_circle_with_border.png')
    green_img_1.save(path_1s + 'green_circle_with_border.png')
    blue_img_1.save(path_1s + 'blue_circle_with_border.png')

    num_r_2 = random.randint(1, num_r_img)
    num_g_2 = random.randint(1, num_b_img)
    num_b_2 = random.randint(1, num_g_img)
    print(num_r_2, num_g_2, num_b_2)
    red_img_2 = Image.open(path_red + str(num_r_2) + '.png')  
    green_img_2 = Image.open(path_green + str(num_g_2) + '.png')  
    blue_img_2 = Image.open(path_blue + str(num_b_2) + '.png')  
    path_2s = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/models/2_stacks/materials/textures/"
    red_img_2.save(path_2s + 'red_circle_with_border.png')
    green_img_2.save(path_2s + 'green_circle_with_border.png')
    blue_img_2.save(path_2s + 'blue_circle_with_border.png')

    num_r_3 = random.randint(1, num_r_img)
    num_g_3 = random.randint(1, num_b_img)
    num_b_3 = random.randint(1, num_g_img)
    print(num_r_3, num_g_3, num_b_3)
    red_img_3 = Image.open(path_red + str(num_r_3) + '.png')  
    green_img_3 = Image.open(path_green + str(num_g_3) + '.png')  
    blue_img_3 = Image.open(path_blue + str(num_b_3) + '.png')  
    path_1_1 = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/models/L-corridor_curve_real_side1_1/materials/textures/"
    red_img_3.save(path_1_1 + 'red_circle_with_border.png')
    green_img_3.save(path_1_1 + 'green_circle_with_border.png')
    blue_img_3.save(path_1_1 + 'blue_circle_with_border.png')

    num_r_4 = random.randint(1, num_r_img)
    num_g_4 = random.randint(1, num_b_img)
    num_b_4 = random.randint(1, num_g_img)
    print(num_r_4, num_g_4, num_b_4)
    red_img_4 = Image.open(path_red + str(num_r_4) + '.png')  
    green_img_4 = Image.open(path_green + str(num_g_4) + '.png')  
    blue_img_4 = Image.open(path_blue + str(num_b_4) + '.png')  
    path_1_2 = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/models/L-corridor_curve_real_side1_2/materials/textures/"
    red_img_4.save(path_1_2 + 'red_circle_with_border.png')
    green_img_4.save(path_1_2 + 'green_circle_with_border.png')
    blue_img_4.save(path_1_2 + 'blue_circle_with_border.png')

    num_r_5 = random.randint(1, num_r_img)
    num_g_5 = random.randint(1, num_b_img)
    num_b_5 = random.randint(1, num_g_img)
    print(num_r_5, num_g_5, num_b_5)
    red_img_5 = Image.open(path_red + str(num_r_5) + '.png')  
    green_img_5 = Image.open(path_green + str(num_g_5) + '.png')  
    blue_img_5 = Image.open(path_blue + str(num_b_5) + '.png')  
    path_2_1 = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/models/L-corridor_curve_real_side2_1/materials/textures/"
    red_img_5.save(path_2_1 + 'red_circle_with_border.png')
    green_img_5.save(path_2_1 + 'green_circle_with_border.png')
    blue_img_5.save(path_2_1 + 'blue_circle_with_border.png')

    num_r_6 = random.randint(1, num_r_img)
    num_g_6 = random.randint(1, num_b_img)
    num_b_6 = random.randint(1, num_g_img)
    print(num_r_6, num_g_6, num_b_6)
    red_img_6 = Image.open(path_red + str(num_r_6) + '.png')  
    green_img_6 = Image.open(path_green + str(num_g_6) + '.png')  
    blue_img_6 = Image.open(path_blue + str(num_b_6) + '.png')  
    path_2_2 = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/models/L-corridor_curve_real_side2_2/materials/textures/"
    red_img_6.save(path_2_2 + 'red_circle_with_border.png')
    green_img_6.save(path_2_2 + 'green_circle_with_border.png')
    blue_img_6.save(path_2_2 + 'blue_circle_with_border.png')

if __name__ == '__main__':
    rospy.init_node('change_box_color_in_env', anonymous=True)
    change_envri()
    