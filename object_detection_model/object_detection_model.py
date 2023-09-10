import tensorflow as tf
import numpy as np
from resnet50 import ExtractFeatures
from region_proposal_network import RegionProposalNetwork
from classification_network import ClassificationNetwork

# Define an Object Detection model
class ObjectDetectionModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        
        # Initialize the ResNet50 model for feature extraction
        self.resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(375, 1242, 3))
        
        # Stage 1: Feature extraction
        self.stage1_feature_extractor = ExtractFeatures()
        
        # Stage 2: Region Proposal Network (RPN)
        self.stage2_region_proposal_network = RegionProposalNetwork()
        
        # Stage 3: Classification Network
        self.stage3_classification_network = ClassificationNetwork(num_classes=num_classes)

    def call(self, images, ground_truth_boxes=None, ground_truth_classes=None):
        # Determine the mode (training or evaluation)
        if ground_truth_boxes is None and ground_truth_classes is None:
            mode = "eval"
        else:
            mode = "train"

        # Extract feature maps from the input images using ResNet50
        feature_maps = self.resnet50(images)

        if mode == "eval":
            # In evaluation mode, generate region proposals and classify them
            proposals = self.stage2_region_proposal_network(feature_maps, ground_truth_boxes)
            classes = self.stage3_classification_network(feature_maps, proposals, ground_truth_classes)
            return classes, proposals

        if mode == "train":
            # In training mode, generate region proposals, calculate losses, and total loss
            proposals, rpn_loss = self.stage2_region_proposal_network(feature_maps, ground_truth_boxes)
            classifier_loss = self.stage3_classification_network(feature_maps, proposals, ground_truth_classes)
            total_loss = rpn_loss + classifier_loss
            return total_loss
