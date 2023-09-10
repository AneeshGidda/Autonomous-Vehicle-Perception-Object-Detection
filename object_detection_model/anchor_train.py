import numpy as np
from anchor_boxes import compute_anchor_sizes, compute_anchor_centers, generate_anchor_boxes
from scores_and_offsets import create_iou_matrix, calculate_offsets
from convert import convert_boxes

def anchor_train(feature_map, all_pred_scores, all_pred_offsets, all_ground_truth_boxes, object_iou_threshold = 0.7):
    batch = np.shape(feature_map)[0]
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    img_shape = [375, 1242, 3]

    all_pred_scores = np.reshape(all_pred_scores, newshape=(batch, -1)) # batch, num of anchors
    all_pred_offsets = np.reshape(all_pred_offsets, newshape=(batch, -1, 4)) # batch, num of anchors, offsets

    final_anchors = []
    final_pred_scores = []
    final_pred_offsets = []
    final_real_scores = []
    final_real_offsets = []

    for pred_offsets, pred_scores, ground_truth_boxes in zip(all_pred_offsets, all_pred_scores, all_ground_truth_boxes):
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
        pred_scores = pred_scores[valid_indices]
        pred_offsets = pred_offsets[valid_indices]

        ground_truth_boxes = convert_boxes(ground_truth_boxes, img_shape, feature_map, mode="image_to_feature")
        iou_matrix = create_iou_matrix(anchor_boxes, ground_truth_boxes)

        max_iou_per_anchor = np.max(iou_matrix, axis=1)
        max_iou_per_anchor_index = np.argmax(iou_matrix, axis=1)
        max_iou_per_ground_truth_box = np.max(iou_matrix, axis=0)
        max_iou_per_ground_truth_box_index = np.where(iou_matrix == max_iou_per_ground_truth_box)[0]

        number_of_anchors = np.shape(anchor_boxes)[0]
        objectness_score = np.full(number_of_anchors, 0)
        objectness_score[max_iou_per_anchor >= object_iou_threshold] = 1
        objectness_score[max_iou_per_ground_truth_box_index] = 1

        positive_anchors_index = np.where(objectness_score == 1)
        pred_scores = pred_scores[positive_anchors_index]
        pred_offsets = pred_offsets[positive_anchors_index]
        real_scores = max_iou_per_anchor[positive_anchors_index]
        positive_anchors = anchor_boxes[positive_anchors_index]
        image_space_positive_anchors = convert_boxes(positive_anchors, img_shape, feature_map, mode="feature_to_image").astype("float32")
        image_space_ground_truth_boxes = convert_boxes(ground_truth_boxes, img_shape, feature_map, mode="feature_to_image")

        corresponding_ground_truth_boxes = image_space_ground_truth_boxes[max_iou_per_anchor_index, :]
        positive_corresponding_ground_truth_boxes = corresponding_ground_truth_boxes[positive_anchors_index].astype("float32")
        real_offsets = calculate_offsets(image_space_positive_anchors, positive_corresponding_ground_truth_boxes)

        final_anchors.append(image_space_positive_anchors)
        final_pred_scores.append(pred_scores)
        final_pred_offsets.append(pred_offsets)
        final_real_scores.append(real_scores)
        final_real_offsets.append(real_offsets)

    return (np.array(final_anchors, dtype="float32"), 
            np.array(final_pred_scores, dtype="float32"), 
            np.array(final_pred_offsets, dtype="float32"),
            np.array(final_real_scores, dtype="float32"), 
            np.array(final_real_offsets, dtype="float32"))
