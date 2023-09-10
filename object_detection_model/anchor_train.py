# Import necessary libraries
import numpy as np
from anchor_boxes import compute_anchor_sizes, compute_anchor_centers, generate_anchor_boxes
from scores_and_offsets import create_iou_matrix, calculate_offsets
from convert import convert_boxes

# Function to train anchor boxes based on feature map, predicted scores, offsets, and ground truth boxes
def anchor_train(feature_map, all_pred_scores, all_pred_offsets, all_ground_truth_boxes, object_iou_threshold=0.7):
    """
    Train anchor boxes based on a feature map, predicted scores, offsets, and ground truth boxes.

    Args:
        feature_map (ndarray): Feature map used for anchor generation.
        all_pred_scores (ndarray): Predicted scores for anchor boxes.
        all_pred_offsets (ndarray): Predicted offsets for anchor boxes.
        all_ground_truth_boxes (ndarray): Ground truth bounding boxes.

    Returns:
        Tuple of ndarrays: Final positive anchor boxes, predicted scores, offsets, real scores, and real offsets.
    """
    # Get batch size, height, and width of the feature map
    batch = np.shape(feature_map)[0]
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    img_shape = [375, 1242, 3]  # Define the shape of the input image

    # Reshape the predicted scores and offsets for easier processing
    all_pred_scores = np.reshape(all_pred_scores, newshape=(batch, -1))
    all_pred_offsets = np.reshape(all_pred_offsets, newshape=(batch, -1, 4))

    # Initialize lists to store final results
    final_anchors = []
    final_pred_scores = []
    final_pred_offsets = []
    final_real_scores = []
    final_real_offsets = []

    # Iterate over batches of predicted offsets, predicted scores, and ground truth boxes
    for pred_offsets, pred_scores, ground_truth_boxes in zip(all_pred_offsets, all_pred_scores, all_ground_truth_boxes):
        base_size = 8
        scales = [1.0, 2.0, 4.0]
        ratios = [0.5, 1.0, 2.0]
    
        # Compute anchor sizes, centers, and generate anchor boxes
        anchor_sizes = compute_anchor_sizes(base_size, scales, ratios)
        anchor_x, anchor_y = compute_anchor_centers(feature_map)
        anchor_boxes = generate_anchor_boxes(anchor_x, anchor_y, anchor_sizes)

        # Filter out anchor boxes outside the image boundaries
        valid_indices = np.where((anchor_boxes[:, 0] > 0) & (anchor_boxes[:, 0] < width) &
                                 (anchor_boxes[:, 1] > 0) & (anchor_boxes[:, 1] < height) &
                                 (anchor_boxes[:, 2] > 0) & (anchor_boxes[:, 2] < width) &
                                 (anchor_boxes[:, 3] > 0) & (anchor_boxes[:, 3] < height))
        
        anchor_boxes = anchor_boxes[valid_indices]
        pred_scores = pred_scores[valid_indices]
        pred_offsets = pred_offsets[valid_indices]

        # Convert ground truth boxes from image space to feature space
        ground_truth_boxes = convert_boxes(ground_truth_boxes, img_shape, feature_map, mode="image_to_feature")
        
        # Create an IoU matrix between anchor boxes and ground truth boxes
        iou_matrix = create_iou_matrix(anchor_boxes, ground_truth_boxes)

        # Calculate various indices and scores for positive anchors
        max_iou_per_anchor = np.max(iou_matrix, axis=1)
        max_iou_per_anchor_index = np.argmax(iou_matrix, axis=1)
        max_iou_per_ground_truth_box = np.max(iou_matrix, axis=0)
        max_iou_per_ground_truth_box_index = np.where(iou_matrix == max_iou_per_ground_truth_box)[0]

        # Initialize objectness score and identify positive anchors
        number_of_anchors = np.shape(anchor_boxes)[0]
        objectness_score = np.full(number_of_anchors, 0)
        objectness_score[max_iou_per_anchor >= object_iou_threshold] = 1
        objectness_score[max_iou_per_ground_truth_box_index] = 1

        # Select positive anchors and convert their coordinates from feature space to image space
        positive_anchors_index = np.where(objectness_score == 1)
        pred_scores = pred_scores[positive_anchors_index]
        pred_offsets = pred_offsets[positive_anchors_index]
        real_scores = max_iou_per_anchor[positive_anchors_index]
        positive_anchors = anchor_boxes[positive_anchors_index]
        image_space_positive_anchors = convert_boxes(positive_anchors, img_shape, feature_map, mode="feature_to_image").astype("float32")
        image_space_ground_truth_boxes = convert_boxes(ground_truth_boxes, img_shape, feature_map, mode="feature_to_image")

        # Get the corresponding ground truth boxes for positive anchors and calculate real offsets
        corresponding_ground_truth_boxes = image_space_ground_truth_boxes[max_iou_per_anchor_index, :]
        positive_corresponding_ground_truth_boxes = corresponding_ground_truth
