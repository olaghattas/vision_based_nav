# Vision_based_navigation_ttt_ml
This package provides a means for a mobile robot equipped with a monocular camera to navigate in unknown environments using a visual quantity called time-to-transit (tau). The package includes code that utilizes computer vision techniques, specifically the Lucas-Kanade method, to estimate time-to-transit by calculating sparse optical flow. Additionally, the package offers an alternative method for computing tau values by employing a Deep Neural Network (DNN)-based technique to predict tau values directly from a couple of successive frames, and it also utilizes lidar to calculate tau values.

Moreover, the package includes a deep learning model that predicts the shape of the path ahead, which further enhances the robot's capability to navigate in an unknown environment.

The diagram of the ROS framework is shown in the figure

<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt_ml/assets/diagram.png"/>

## How to Run the Package in Simulation
### Launch All nodes at once:

   **Steer the robot using CNN**
  ```
    roslaunch vision_based_navigation_ttt_ml CNN_launch_file.launch
  ```
   **Steer the robot using Lidar**
  ```
    roslaunch vision_based_navigation_ttt_ml lidar_launch_file.launch
  ```
   **Steer the robot using Velodyne**
  ```
    roslaunch vision_based_navigation_ttt_ml velodyne_launch_file.launch
  ```  
   **Steer the robot using Optical Flow (Computed using the Lukas Kanade method)**
  ```
    roslaunch vision_based_navigation_ttt_ml optical_flow_nodes.launch
  ```

### Run Each Node Separately:

**1. You will need to launch Gazebo first by running**
``` 
roslaunch vision_based_navigation_ttt_ml <your chosen file from launch folders>.launch 
```
To simulate your desired world specify it in the launch file at line: 

  ```
  arg name="world_name" value="$(find vision_based_navigation_ttt_ml)/GazeboWorlds/<files with your desired world found in GazeboWorlds folder>.world" 
  ```

**2. You will also need to calculate the tau values using one of the options available**
  
To get tau values from optical flow run: 

  ```
  rosrun vision_based_navigation_ttt_ml optical_flow.py
  rosrun vision_based_navigation_ttt_ml tau_computation.py 
  ```

 To get tau values from velodyne(3D lidar) run:

  ```
  rosrun vision_based_navigation_ttt_ml tau_computation_velodyne.py 
  ```
  To get tau values from lidar(2D lidar) run:

  ```
  rosrun vision_based_navigation_ttt_ml tau_computation_lidar.py 
  ```

  To get tau values from CNN model run:

  ```
  rosrun vision_based_navigation_ttt_ml tau_computation_cnn.py
  ```
  This window will show two values the top one is the cnn prediction and thebottom one is from the lidar. You can choose the parameters that you want that are available in the trained_model_parameters folder just change the model name in line tf.keras.models.load_model. Not that there are models that take velocities as input and others dont so make sure to choose the function that calculates tau values according tothe model you chose.

**3. Finally, you will also need to run a controller in a seperate terminal with one of the available controllers :**
  
  To use the controller with sense and act phases, run 

  ```
  rosrun vision_based_navigation_ttt_ml controller.py 
  ```

  To use the controller with only an act phase, run 

  ```
  rosrun vision_based_navigation_ttt_ml controller_act_bias.py 
  ```
## How to Run the Package on the Jackal
You need to run the following launch file 
```
  roslaunch vision_based_navigation_ttt_ml CNN_launch_file_on_real_jackal.launch
```
Note that you need to have the realsense-ros package installed. It can be found in https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy.

## Custom Worlds 
Multiple custom worlds were created in Gazebo to resemble the environment being tested on in the lab. 
 
<table border="0">
 <tr>
    <td><b style="font-size:30px">T_shaped corridor</b></td>
    <td><b style="font-size:30px">L_shaped corridor</b></td>
    <td><b style="font-size:30px">U_shaped corridor</b></td>
    <td><b style="font-size:30px">House Garden</b></td>
 </tr>
 <tr>
    <td>
<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt_ml/assets/T_shaped.png"/> 
     </td>
     <td>
<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt_ml/assets/L_shaped.png"/>
      </td>
     <td>
<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt_ml/assets/U_shaped.png"/>
      </td>
      <td>
<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt_ml/assets/House_garden.png"/>
      </td>
 </tr>
</table>

You can build up custom gazebo worlds by using wall segments. The video below shows how this can be done.

<img src="https://user-images.githubusercontent.com/98136555/185213284-8d2cfa97-f4ec-4a5c-a24f-7408b699c902.mp4" width=50% height=50%/>


<!-- ### Performance
  Peformance can be affected by lighting as shown in the videos below.
  
  <table border="0">
 <tr>
    <td><b style="font-size:30px">Two lights</b></td>
    <td><b style="font-size:30px">Three lights</b></td>
 </tr>
 <tr>
    <td>
 
https://user-images.githubusercontent.com/98136555/185210652-f371b74c-7054-4f63-95b3-365b9713b741.mp4</td>
    <td>
      
https://user-images.githubusercontent.com/98136555/185215284-977937bf-bd99-4416-b706-a4d0d101c430.mp4</td>
 </tr>
</table> -->

 
## CNN-Based τ Predicition

The aim is to introduce a Convolutional Neural Network (DNN) that automatically estimates values of tau in the 5 regions of interests from a couple of images, without explicitly computing optical flow. It is reasonable to think that this network learns a form of optical flow in an unsupervised manner through its hidden layers.

  ### Data Collection
 
  To train the CNN, two consecutive images and the corresponding tau values in the respective regions of the images are required. The data_collection.py file is utilized for saving the images and tau values. The tau values are obtained from depth measurements using a lidar, as this provides the most reliable method for obtaining time-to-transit values. The node requires /image_raw and /tau_values topics to receive the required data.
  
  
  To collect data in simulation using a 2D lidar, run the commands below:
  ```
  roslaunch vision_based_navigation_ttt_ml <name-of-launch-file>.launch front_laser:=1
  rosrun vision_based_navigation_ttt_ml tau_computattion_lidar.py 
  rosrun vision_based_navigation_ttt_ml controller.py 
  rosrun vision_based_navigation_ttt_ml data_collection.py 
  ```

  By default, the images will be saved in the ```training_images``` folder and the distances are saved in the ```tau_values``` folder. 
  
  ### Available Model Architectures to Train :
  
  #### 1. cnn_auto_ml
  This model uses "AutoKeras" which is an AutoML System. It takes two successive colored images as input, and outputs the distance in each region of interest. The distance is then converted to ```tau_value``` by dividing it by the robot's velocity.

 ##### Demo:
 <table border="0">
 <tr>
    <td><b style="font-size:30px">Model ran in an environment it was not trained on</b></td>
    <td><b style="font-size:30px">Model ran in a T-shaped corridor</b></td>
 </tr>
 <tr>
    <td>

https://user-images.githubusercontent.com/98136555/211264448-130d28b4-0fb9-4551-9ef9-4cc48a1fa0b1.mp4

 </td>
    <td>

https://user-images.githubusercontent.com/98136555/211263011-e2469251-4f1f-49e2-b989-e46dfc45e910.mp4
  </td>
 </tr>
</table> 
  
  ##### Model Architecture:
  
  <img src="https://user-images.githubusercontent.com/98136555/211239897-3d31f95e-03bc-45ba-96e7-9a65a0e81cef.png" width=25% height=25%/>
  
  
  #### 2. cnn_colored_output_distance_in_each_roi
This model takes two colored images as input, and outputs an array that contains the distance in each roi.
   ##### Demo:

   <table border="0">
 <tr>
    <td><b style="font-size:30px">Model ran in an environment it was trained on</b></td>
    <td><b style="font-size:30px">Model ran in a T-shaped corridor</b></td>
 </tr>
 <tr>
    <td>

https://user-images.githubusercontent.com/98136555/211262755-43a8d499-1b23-40f4-a373-ea8c67d1b607.mp4

 </td>
    <td>

https://user-images.githubusercontent.com/98136555/211262738-a77bb3e2-d42a-404e-9bba-cd417e688f82.mp4

  </td>
 </tr>
</table>

   ##### Model Architecture:
   
  <img src="https://user-images.githubusercontent.com/98136555/211247640-d3bb4dd1-b210-4fbd-adc4-8059609093ae.png" width=25% height=25%/>

  #### 3. cnn_grayscale_output_tau_value_in_each_roi
  This model takes two grayscale images and the velocity as input, and outputs an array that contains the ```tau_values``` in each roi.
  
   ##### Model Architecture:
   
   <img src="https://user-images.githubusercontent.com/98136555/211253489-fc6b081e-af00-4c99-a85f-3cd9153b509c.png" width=25% height=25%/>

  #### 4. cnn_output_tau_value_in_each_roi_and_validity
  
  The model takes two successive images along with the velocity as input, and outputs two arrays one contains the tau values in each region of interest , and the other contains a flag that shows if the predicited value is valid or not.
  
  ##### Model Architecture:
  
  <img src="https://user-images.githubusercontent.com/98136555/203196927-e1a5df6a-b659-4cb5-899a-96971d8fb24e.png" width=25% height=25%/>

  #### 5. cnn_input_roi_image
  
  Unlike the previous models, this model takes two successive images of the region of interest as input, and outputs the tau value in that region. This model is computationally expensive since it has to run 5 times to get the tau value for the 5 regions of interest.
  
   ##### Model Architecture:
   
 <img src="https://user-images.githubusercontent.com/98136555/203196713-d184d217-4d4c-4703-9a3e-b70578cf4f85.png" width=25% height=25%/>
  
  ### Collected data and trained models:
  The Dataset collected to train the models can be found in https://drive.google.com/drive/folders/14Z0PIDKhXRiH8N9Lk4W1LmMqiyPlzIl8?usp=share_link. The folder also includes a Readme file that explains what each folder includes.
  
  The parameters for the trained models can be found in https://drive.google.com/drive/folders/1mN2qUArRAUh9lco24jHvgxeGaYj4N5pD?usp=share_link.
  
  ## CNN-Based Turn Detection:
  For this we trained several well-known architectures and assessed their performance, and the ResNet50v2 architecture demonstrated the highest performance.
  
  The file reponsible for this is publish_shape_corridor.py, that subscribers to the image topic and produces a vector of three binary elements, indicating the presence of a left, right, or straight path respectively. 
  
  The dataset required to train this model can be collected using the automatic_label_sim.py file. It's worth noting that this function is currently only compatible with a limited number of predefined environments. If you need to add another environment, you will have to map the x, y, and theta coordinates of the robot to determine what turns are visible from that position. Another way to collect the dataset is to label the images manually using the GUI available in the manual_label_turns.py file.
  
  ## Lab Results:
  For predicting tau values, the model employed is based on the resnetv2-101 architecture, which was trained on data gathered from the real robot. On the other hand, to predict the shape of the corridor, specifically the upcoming turns visible in the image, a model based on the resnetv2-50 architecture is used. This model was trained using simulated and real data. The models used were selected based on their superior performance.
  
  <img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt_ml/assets/IROS23_lab_exp.webm"/>
