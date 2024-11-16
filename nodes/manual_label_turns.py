import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
import csv
import os

class ImageViewer(QWidget):
    def __init__(self, tuple_):
        super().__init__()
        self.tuple_ = tuple_
        # Create a layout for the widgets
        self.layout = QVBoxLayout(self)

        # Create a label to display the image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        self.layout.addLayout(button_layout)

        # Create buttons
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_image)
        button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)
        button_layout.addWidget(self.next_button)

        self.prev_fast_button = QPushButton("Previous_fast")
        self.prev_fast_button.clicked.connect(self.show_previous_10_image)
        button_layout.addWidget(self.prev_fast_button)

        self.next_fast_button = QPushButton("Next_fast")
        self.next_fast_button.clicked.connect(self.show_next_10_image)
        button_layout.addWidget(self.next_fast_button)

        self.left_button = QPushButton("Left")
        self.left_button.clicked.connect(self.label_left)
        button_layout.addWidget(self.left_button)

        self.right_button = QPushButton("Right")
        self.right_button.clicked.connect(self.label_right)
        button_layout.addWidget(self.right_button)

        self.straight_button = QPushButton("Straight")
        self.straight_button.clicked.connect(self.label_straight)
        button_layout.addWidget(self.straight_button)

        self.left_right_button = QPushButton("Left_Right")
        self.left_right_button.clicked.connect(self.label_left_right)
        button_layout.addWidget(self.left_right_button)

        self.straight_left_button = QPushButton("Straight_Left")
        self.straight_left_button.clicked.connect(self.label_straight_left)
        button_layout.addWidget(self.straight_left_button)

        self.straight_left_right_button = QPushButton("Straight_Left_Right")
        self.straight_left_right_button.clicked.connect(self.label_straight_left_right)
        button_layout.addWidget(self.straight_left_right_button)

        self.straight_right_button = QPushButton("Straight_Right")
        self.straight_right_button.clicked.connect(self.label_straight_right)
        button_layout.addWidget(self.straight_right_button)

        self.images = []
        self.current_image = 0 

        # load the dictionary
        self.csv_file_name = 'labels_file_tr.csv'
        self.path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/"
        self.dict_labels = read_from_csv(self.path + self.csv_file_name)
        self.count = 0

    def loop_images(self):
        # Set the images to display
        print('here')
        path_images_folder = self.path + "olaf_img/"
        folders = [file for file in sorted(os.listdir(path_images_folder)) if os.path.isdir(os.path.join(path_images_folder, file))]
        
        # for folder in folders:
        folder = folders[7] #0,

        print(folder)
        path_images = path_images_folder + folder + '/'
        images_in_folder = [path_images + f for f in sorted(os.listdir(path_images)) if f.endswith(".png")]
        self.images += images_in_folder
        print('ci',self.current_image)
        if self.tuple_:
            # retreive direction
            self.count += 1
            print(self.count , str(self.images[self.current_image].split('/')[8].split('.')[0]))
            label = self.dict_labels[str(self.images[self.current_image].split('/')[8].split('.')[0])]
            print('type',type(label))
            tuple_to_display = eval(label)
            print('t',tuple_to_display)
            print('ci',self.current_image)
            if self.current_image ==0:
                self.label_ = QLabel('Tuple: {}'.format(tuple_to_display))  
                self.layout.addWidget(self.label_)
            else:
                self.label_.setText('Tuple: {}'.format(tuple_to_display))
        # Show the first image
        self.show_image()

    def show_previous_image(self):
        self.current_image -= 1
        if self.current_image < 0:
            self.current_image = len(self.images) - 1
        self.show_image()
        self.loop_images()

    def show_next_image(self):
        self.current_image += 1
        if self.current_image >= len(self.images):
            self.current_image = 0
        self.show_image()
        self.loop_images()

    def show_previous_10_image(self):
        self.current_image -= 10
        if self.current_image < 0:
            self.current_image = len(self.images) - 1
        self.show_image()
        self.loop_images()

    def show_next_10_image(self):
        self.current_image += 10
        if self.current_image >= len(self.images):
            self.current_image = 0
        self.show_image()
        self.loop_images()
    
    def label_left_right(self):
        self.dict_labels[self.images[self.current_image].split('/')[8].split('.')[0]] = '(1,1,0)'
        # print('img_num',self.images[self.current_image].split('/')[8].split('.')[0])
        # print('dict',self.dict_labels)
        save_dict_to_csv(self.dict_labels, self.csv_file_name)
        self.show_next_image()

    def label_left(self):
        self.dict_labels[self.images[self.current_image].split('/')[8].split('.')[0]] = '(1,0,0)'
        # print('img_num',self.images[self.current_image].split('/')[8].split('.')[0])
        # print('dict',self.dict_labels)
        save_dict_to_csv(self.dict_labels, self.csv_file_name)
        self.show_next_image()
    
    def label_right(self):
        self.dict_labels[self.images[self.current_image].split('/')[8].split('.')[0]] = '(0,1,0)'
        # print('img_num',self.images[self.current_image].split('/')[8].split('.')[0])
        # print('dict',self.dict_labels)
        save_dict_to_csv(self.dict_labels, self.csv_file_name)
        self.show_next_image()

    def label_straight(self):
        self.dict_labels[self.images[self.current_image].split('/')[8].split('.')[0]] = '(0,0,0)'
        # print('img_num',self.images[self.current_image].split('/')[8].split('.')[0])
        # print('dict',self.dict_labels)
        img_num = self.images[self.current_image].split('/')[8].split('.')[0]
        print('img_num',img_num)
        print('dict',self.dict_labels[img_num])
        save_dict_to_csv(self.dict_labels, self.csv_file_name)
        self.show_next_image()

    def label_straight_left_right(self):
        self.dict_labels[self.images[self.current_image].split('/')[8].split('.')[0]] = '(1,1,1)'
        # print('img_num',self.images[self.current_image].split('/')[8].split('.')[0])
        # print('dict',self.dict_labels)
        save_dict_to_csv(self.dict_labels, self.csv_file_name)
        self.show_next_image()

    def label_straight_left(self):
        self.dict_labels[self.images[self.current_image].split('/')[8].split('.')[0]] = '(1,0,1)'
        img_num = self.images[self.current_image].split('/')[8].split('.')[0]
        print('img_num',img_num)
        print('dict',self.dict_labels[img_num])
        save_dict_to_csv(self.dict_labels, self.csv_file_name)
        self.show_next_image()

    def label_straight_right(self):
        self.dict_labels[self.images[self.current_image].split('/')[8].split('.')[0]] = '(0,1,1)'
        # print('img_num',self.images[self.current_image].split('/')[8].split('.')[0])
        # print('dict',self.dict_labels)
        save_dict_to_csv(self.dict_labels, self.csv_file_name)
        self.show_next_image()

    def show_image(self):
        image = QPixmap(self.images[self.current_image])
        self.image_label.setPixmap(image)

def save_dict_to_csv(dictionary, dict_name):
    name = dict_name
    path = os.environ["HOME"]+"/catkin_ws/src/vision_based_navigation_ttt_ml/"
    # Open the file in writing mode (no blank lines)
    with open(path + name, 'w', newline='') as f:
        # Create a CSV writer object
        writer = csv.writer(f)
        # Write one key-value tuple per row
        for row in dictionary.items():
            writer.writerow(row)

def read_from_csv(csv_filename):
    dict_from_csv = {}

    with open(csv_filename, mode='r') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[0]:rows[1] for rows in reader}
    return dict_from_csv

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer(0)
    viewer.show()
    viewer.loop_images()
    # save_dict_to_csv(viewer.dict_labels)
    sys.exit(app.exec_())
    
    
