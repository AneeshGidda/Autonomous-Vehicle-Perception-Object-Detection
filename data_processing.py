import os
import cv2
import numpy as np
import pandas as pd

image_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images\training" 
label_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels"
lidar_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_lidar\training"

height_list = []
width_list = []
img_list = []

for root, dirs, files in os.walk(image_directory):
    for file in files[:3]:
        img_path = os.path.join(root, file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_list.append(img)

for img in img_list:
    height_list.append(np.shape(img)[0])
    width_list.append(np.shape(img)[1])

height = max(height_list)
width = max(width_list)

for i, img in enumerate(img_list):
    height_diff = height - np.shape(img)[0]
    width_diff = width - np.shape(img)[1]
    img_list[i] = np.pad(img, ((0, height_diff), (0, width_diff), (0, 0)), mode='constant')

max_boxes_per_file = 22
class_list = {"Car", "Pedestrian", "Truck", "Van", "Tram", "Cyclist", "Misc", "DontCare", "0"}
ground_truth_boxes = []
ground_truth_classes = []
for root, dirs, files in os.walk(label_directory):
    for file in files[0:3]:
        file_path = os.path.join(root, file)
        boxes = []
        classes = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split()
                boxes.append([data[4], data[5], data[6], data[7]])
                classes.append(data[0])
            while len(boxes) < max_boxes_per_file:
                boxes.append([0, 0, 0, 0])
                classes.append('0')
        one_hot_encoded = pd.get_dummies(classes, columns=class_list)
        for class_name in class_list:
            if class_name not in one_hot_encoded.columns:
                one_hot_encoded[class_name] = 0
        classes = one_hot_encoded[sorted(class_list)]
        ground_truth_boxes.append(boxes)
        ground_truth_classes.append(classes)

images = np.stack(img_list, axis=0)
ground_truth_boxes = np.array(ground_truth_boxes, dtype="float32")
ground_truth_classes = np.array(ground_truth_classes, dtype="int32")
print(np.shape(images))
print(np.shape(ground_truth_boxes))
print(np.shape(ground_truth_classes))

np.save("images.npy", images)
np.save("ground_truth_boxes.npy", ground_truth_boxes)
np.save("ground_truth_classes.npy", ground_truth_classes)

# print(f"shape of x_data: {x_data_shape}")
# print(f"sample image:\n{x_data[0]}")

# for root, dirs, files in os.walk(lidar_directory):
#     for file in files[4:5]:
#         file_path = os.path.join(root, file)
#         lidar_data = np.fromfile(file_path, dtype=np.float32)

#         lidar_data = lidar_data.reshape((-1, 4))
#         lidar_data_shape = np.shape(lidar_data)
#         print(f"shape of lidar data: {lidar_data_shape}")
#         x = lidar_data[:, 0]
#         y = lidar_data[:, 1]
#         z = lidar_data[:, 2] 
#         intensity = lidar_data[:, 3]  
#         for i in range(len(lidar_data)):
#             print(f"Point {i+1}: x={x[i]}, y={y[i]}, z={z[i]}, intensity={intensity[i]}")
