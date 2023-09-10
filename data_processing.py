# Import necessary libraries
import os
import cv2
import numpy as np
import pandas as pd

# Define directories where your data is stored
image_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images\training"
label_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels"
lidar_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_lidar\training"

# Lists to store image data
height_list = []
width_list = []
img_list = []

# Loop through image files in the specified directory
for root, dirs, files in os.walk(image_directory):
    for file in files:
        img_path = os.path.join(root, file)
        
        # Read image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_list.append(img)

# Find the maximum height and width among the images
height = max(height_list)
width = max(width_list)

# Resize images to have uniform dimensions
for i, img in enumerate(img_list):
    height_diff = height - np.shape(img)[0]
    width_diff = width - np.shape(img)[1]
    img_list[i] = np.pad(img, ((0, height_diff), (0, width_diff), (0, 0)), mode='constant')

# Define constants
max_boxes_per_file = 22
class_list = {"Car", "Pedestrian", "Truck", "Van", "Tram", "Cyclist", "Misc", "DontCare", "0"}

# Lists to store ground truth data
ground_truth_boxes = []
ground_truth_classes = []

# Loop through label files in the specified directory
for root, dirs, files in os.walk(label_directory):
    for file in files:
        file_path = os.path.join(root, file)
        boxes = []
        classes = []
        
        # Read and process label data
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split()
                boxes.append([data[4], data[5], data[6], data[7]])
                classes.append(data[0])
            
            # Pad the data to reach the maximum number of boxes
            while len(boxes) < max_boxes_per_file:
                boxes.append([0, 0, 0, 0])
                classes.append('0')
        
        # One-hot encode the class labels
        one_hot_encoded = pd.get_dummies(classes, columns=class_list)
        
        # Ensure all class columns are present (filled with 0s if not)
        for class_name in class_list:
            if class_name not in one_hot_encoded.columns:
                one_hot_encoded[class_name] = 0
        
        # Sort the class columns
        classes = one_hot_encoded[sorted(class_list)]
        
        # Append data to the respective lists
        ground_truth_boxes.append(boxes)
        ground_truth_classes.append(classes)

# Convert data to NumPy arrays
images = np.stack(img_list, axis=0)
ground_truth_boxes = np.array(ground_truth_boxes, dtype="float32")
ground_truth_classes = np.array(ground_truth_classes, dtype="int32")

# Save the data as NumPy files
np.save("images.npy", images)
np.save("ground_truth_boxes.npy", ground_truth_boxes)
np.save("ground_truth_classes.npy", ground_truth_classes)
