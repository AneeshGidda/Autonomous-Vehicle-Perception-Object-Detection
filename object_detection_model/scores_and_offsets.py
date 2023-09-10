import numpy as np
import tensorflow as tf
from convert import convert_format

def intersection_over_matrix(anchor_box, ground_truth_box):
    """
    Computes the Intersection over Union (IoU) between an anchor box and a ground truth box.

    Args:
        anchor_box (list): A bounding box represented as [x_min, y_min, x_max, y_max].
        ground_truth_box (list): A ground truth bounding box represented as [x_min, y_min, x_max, y_max].

    Returns:
        float: The IoU value between the two input boxes.
    """
    # Compute the coordinates of the intersection rectangle
    x1 = np.maximum(anchor_box[0], ground_truth_box[0])
    y1 = np.maximum(anchor_box[1], ground_truth_box[1])
    x2 = np.minimum(anchor_box[2], ground_truth_box[2])
    y2 = np.minimum(anchor_box[3], ground_truth_box[3])

    # Calculate the area of intersection and the areas of the anchor and ground truth boxes
    intersection_area = max(0, (y2 - y1)) * max(0, (x2 - x1))
    area_1 = (anchor_box[2] - anchor_box[0]) * (anchor_box[3] - anchor_box[1])
    area_2 = (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])

    # Calculate the intersection over union (IoU)
    union_area = area_1 + area_2 - intersection_area
    iou = intersection_area / union_area
    return iou

def create_iou_matrix(anchor_boxes, ground_truth_boxes):
    iou_matrix = []
    for anchor_box in anchor_boxes:
        anchor_box_iou = []
        for ground_truth_box in ground_truth_boxes:
            # Calculate IoU for each anchor box with each ground truth box
            iou = intersection_over_matrix(anchor_box, ground_truth_box)
            anchor_box_iou.append(iou)
        iou_matrix.append(anchor_box_iou)
    return np.array(iou_matrix)

def calculate_offsets(positive_anchors, ground_truth_boxes):
    # Convert the box formats to (cx, cy, w, h) representation
    positive_anchors = convert_format(positive_anchors, mode="xyxy_to_cxcywh")
    ground_truth_boxes = convert_format(ground_truth_boxes, mode="xyxy_to_cxcywh")

    # Unstack the coordinates for anchors and ground truth boxes
    gt_cx, gt_cy, gt_w, gt_h = tf.unstack(ground_truth_boxes, axis=1)
    anc_cx, anc_cy, anc_w, anc_h = tf.unstack(positive_anchors, axis=1)

    # Calculate the offset values (tx, ty, tw, th)
    tx = (gt_cx - anc_cx) / anc_w
    ty = (gt_cy - anc_cy) / anc_h
    tw = tf.math.log(gt_w / anc_w)
    th = tf.math.log(gt_h / anc_h)
    
    # Stack the offset values into a single tensor
    return tf.stack([tx, ty, tw, th], axis=-1)
