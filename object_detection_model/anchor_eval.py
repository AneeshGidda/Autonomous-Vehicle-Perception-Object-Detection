import numpy as np
from anchor_boxes import compute_anchor_sizes, compute_anchor_centers, generate_anchor_boxes
from convert import convert_boxes

def anchors_eval(feature_map, all_scores, all_offsets):
    batch = np.shape(feature_map)[0]
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    img_shape = [375, 1242, 3]

    all_scores = np.reshape(all_scores, newshape=(batch, -1)) # batch, num of anchors
    all_offsets = np.reshape(all_offsets, newshape=(batch, -1, 4)) # batch, num of anchors, offsets

    final_anchors = []
    final_scores = []
    final_offsets = []

    for offsets, scores in zip(all_offsets, all_scores):
        base_size = 8
        scales = [1.0, 2.0, 4.0]
        ratios = [0.5, 1.0, 2.0]
    
        anchor_sizes = compute_anchor_sizes(base_size, scales, ratios)
        anchor_x, anchor_y = compute_anchor_centers(feature_map)
        anchor_boxes = generate_anchor_boxes(anchor_x, anchor_y, anchor_sizes)

        valid_indices = np.where((anchor_boxes[:, 0] > 0) & (anchor_boxes[:, 0] < width) &
                             (anchor_boxes[:, 1] > 0) & (anchor_boxes[:, 1] < height) &
                             (anchor_boxes[:, 2] > 0) & (anchor_boxes[:, 2] < width) &
                             (anchor_boxes[:, 3] > 0) & (anchor_boxes[:, 3] < height))
        
        anchor_boxes = anchor_boxes[valid_indices]
        scores = scores[valid_indices]
        offsets = offsets[valid_indices]

        number_of_anchors = np.shape(anchor_boxes)[0]
        anchors_mask = np.full(number_of_anchors, 0)
        anchors_mask[scores >= 0.7] = 1

        positive_anchors_indices = np.squeeze(np.where(anchors_mask == 1))
        positive_anchors = anchor_boxes[positive_anchors_indices]
        positive_anchors = convert_boxes(positive_anchors, img_shape, feature_map, mode="feature_to_image")
        scores = scores[positive_anchors_indices]
        offsets = offsets[positive_anchors_indices]

        final_anchors.append(positive_anchors)
        final_scores.append(scores)
        final_offsets.append(offsets)
    return np.array(final_anchors, dtype="float32"), np.array(final_scores, dtype="float32"), np.array(final_offsets, dtype="float32")
