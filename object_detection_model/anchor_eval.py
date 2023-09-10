# Import necessary libraries
import numpy as np
from anchor_boxes import compute_anchor_sizes, compute_anchor_centers, generate_anchor_boxes
from convert import convert_boxes

# Function to evaluate anchor boxes based on feature map, scores, and offsets
def anchors_eval(feature_map, all_scores, all_offsets):
    """
    Evaluate anchor boxes based on a feature map, scores, and offsets.

    Args:
        feature_map (ndarray): Feature map used for anchor generation.
        all_scores (ndarray): All anchor scores.
        all_offsets (ndarray): All anchor offsets.

    Returns:
        Tuple of ndarrays: Final positive anchor boxes, their scores, and offsets.
    """
    # Get the batch size, height, and width of the feature map
    batch = np.shape(feature_map)[0]
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    img_shape = [375, 1242, 3]  # Define the shape of the input image

    # Reshape the scores and offsets for easier processing
    all_scores = np.reshape(all_scores, newshape=(batch, -1))
    all_offsets = np.reshape(all_offsets, newshape=(batch, -1, 4))

    # Initialize lists to store final results
    final_anchors = []
    final_scores = []
    final_offsets = []

    # Iterate over batches of scores and offsets
    for offsets, scores in zip(all_offsets, all_scores):
        # Define anchor box parameters
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
        scores = scores[valid_indices]
        offsets = offsets[valid_indices]

        # Define a mask to identify positive anchors
        number_of_anchors = np.shape(anchor_boxes)[0]
        anchors_mask = np.full(number_of_anchors, 0)
        anchors_mask[scores >= 0.7] = 1

        # Select positive anchors and convert their coordinates
        positive_anchors_indices = np.squeeze(np.where(anchors_mask == 1))
        positive_anchors = anchor_boxes[positive_anchors_indices]
        positive_anchors = convert_boxes(positive_anchors, img_shape, feature_map, mode="feature_to_image")
        scores = scores[positive_anchors_indices]
        offsets = offsets[positive_anchors_indices]

        # Append positive anchors, scores, and offsets to the final lists
        final_anchors.append(positive_anchors)
        final_scores.append(scores)
        final_offsets.append(offsets)
    
    return (
        np.array(final_anchors, dtype="float32"),
        np.array(final_scores, dtype="float32"),
        np.array(final_offsets, dtype="float32")
    )
