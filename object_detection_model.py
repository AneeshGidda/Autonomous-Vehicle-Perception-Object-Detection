import tensorflow as tf
import numpy as np
from resnet50 import ExtractFeatures
from region_proposal_network import RegionProposalNetwork
from classification_network import ClassificationNetwork
import pandas as pd
import cv2

class ObjectDetectionModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(375, 1242, 3))
        self.stage1_feature_extractor = ExtractFeatures()
        self.stage2_region_proposal_network = RegionProposalNetwork()
        self.stage3_classification_network = ClassificationNetwork(num_classes=num_classes)

    def call(self, images, ground_truth_boxes=None, ground_truth_classes=None):
        if ground_truth_boxes is None and ground_truth_classes is None:
            mode = "eval"
        else:
            mode = "train"

        feature_maps = self.resnet50(images)

        if mode == "eval":
            proposals = self.stage2_region_proposal_network(feature_maps, ground_truth_boxes)
            classes = self.stage3_classification_network(feature_maps, proposals, ground_truth_classes)
            return classes, proposals

        if mode == "train":
            proposals, rpn_loss = self.stage2_region_proposal_network(feature_maps, ground_truth_boxes)
            classifier_loss = self.stage3_classification_network(feature_maps, proposals, ground_truth_classes)
            total_loss = rpn_loss + classifier_loss
            return total_loss
    














    
# img_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images\training\image_2\000008.png"
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img = tf.cast(tf.reshape(img, shape=(1, 375, 1242, 3)), dtype="float64")
# label_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels\training\label_2\000008.txt"

# ground_truth_boxes = []
# ground_truth_classes = []

# with open(label_path, 'r') as file:
#     for line in file:
#         data = line.split()
#         ground_truth_boxes.append([data[4] , data[5], data[6], data[7]])
#         ground_truth_classes.append(data[0])

# num_classes = 2
# ground_truth_boxes = np.array(ground_truth_boxes, dtype="float32")
# ground_truth_classes = pd.get_dummies(ground_truth_classes).values

# ground_truth_boxes = np.reshape(ground_truth_boxes, newshape=(1, 10, 4)).astype("float32")
# ground_truth_classes = np.reshape(ground_truth_classes, newshape=(1, 10, 2)).astype("float32")

# model = ObjectDetectionModel(num_classes=num_classes)
# loss = model(img, ground_truth_boxes, ground_truth_classes)
# print(loss)

# final_classes, final_box_deltas = model(img)
# print("Final Classes:")
# for final_class in final_classes:
#     print(final_class)
# print("Final Box Deltas:")
# for final_box_delta in final_box_deltas:
#     print(final_box_delta)