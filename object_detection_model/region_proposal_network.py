import tensorflow as tf
import numpy as np
import cv2
from anchor_train import anchor_train
from anchor_eval import anchors_eval
from convert import convert_proposals
from loss_functions import calc_offset_loss, calc_score_loss
from proposals import generate_proposals, filter_proposals

# Define a Region Proposal Network (RPN) model
class RegionProposalNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.num_of_anchors = 9
        
        # 1st Convolutional layer for feature extraction
        self.conv1 = tf.keras.layers.Conv2D(filters=2048, kernel_size=(3,3), padding="same", activation="relu", trainable=True)
        
        # Classifier layer for anchor classification
        self.classifier = tf.keras.layers.Conv2D(filters=self.num_of_anchors, kernel_size=(1,1), padding="same", activation="sigmoid", trainable=True)
        
        # Regressor layer for anchor regression
        self.regressor = tf.keras.layers.Conv2D(filters=4 * self.num_of_anchors, kernel_size=(1,1), padding="same", activation="linear", trainable=True)

    def call(self, feature_map, ground_truth_boxes=None):
        # Determine the mode (training or evaluation)
        if ground_truth_boxes is None:
            mode = "eval"
        else:
            mode = "train"

        # Extract features from the input feature map
        x = self.conv1(feature_map)
        
        # Generate anchor scores and offsets
        scores = self.classifier(x)
        offsets = self.regressor(x)

        if mode == "eval":
            # In evaluation mode, generate anchors and evaluate them
            anchors, pred_scores, pred_offsets = anchors_eval(feature_map, scores, offsets)

        if mode == "train":
            # In training mode, generate anchors, calculate losses, and filter proposals
            anchors, pred_scores, pred_offsets, real_scores, real_offsets = anchor_train(feature_map, scores, offsets, ground_truth_boxes)
            classifier_loss = calc_score_loss(pred_scores, real_scores)
            offset_loss = calc_offset_loss(pred_offsets, real_offsets)
            total_loss = classifier_loss + offset_loss

        # Generate proposals based on anchors and predicted offsets
        proposals = generate_proposals(anchors, pred_offsets)
        num_ground_truth_boxes = np.shape(ground_truth_boxes)[1]
        num_proposals = max(num_ground_truth_boxes, np.shape(proposals)[1])
        
        # Filter proposals to keep the top ones using NMS
        proposals, scores = filter_proposals(proposals, pred_scores, number_of_proposals=num_proposals)
        nms_proposals = convert_proposals(proposals)

        final_proposals = []
        for proposals, scores in zip(nms_proposals, pred_scores):
            # Apply Non-Maximum Suppression (NMS) to further filter proposals
            indices = tf.image.non_max_suppression(boxes=proposals, max_output_size=num_ground_truth_boxes, scores=scores, iou_threshold=0.7)
            proposals = tf.gather(proposals, indices=indices)
            final_proposals.append(proposals)
        
        if mode == "eval":
            return final_proposals
        
        if mode == "train":
            return final_proposals, total_loss
