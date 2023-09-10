import tensorflow as tf
import numpy as np

def convert_format(boxes, mode):
    """
    Convert bounding box format between 'xyxy' and 'cxcywh'.

    Args:
        boxes (Tensor): Bounding boxes in the original format.
        mode (str): Conversion mode, either 'xyxy_to_cxcywh' or 'cxcywh_to_xyxy'.

    Returns:
        Tensor: Bounding boxes in the new format.
    """
    if mode == "xyxy_to_cxcywh":
        x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=-1)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        cxcywh_boxes = tf.concat([cx, cy, width, height], axis=-1)
        return cxcywh_boxes
    elif mode == "cxcywh_to_xyxy":
        cx, cy, width, height = tf.split(boxes, 4, axis=-1)
        x_min = cx - (width / 2)
        y_min = cy - (height / 2)
        x_max = cx + (width / 2)
        y_max = cy + (height / 2)
        xyxy_boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
        return xyxy_boxes

def convert_boxes(boxes, img_shape, feature_map, mode):
    """
    Convert bounding boxes between image and feature map coordinates.

    Args:
        boxes (ndarray): Bounding boxes in the original format.
        img_shape (list): Shape of the input image [height, width, channels].
        feature_map (Tensor): Feature map used for coordinate conversion.
        mode (str): Conversion mode, either 'image_to_feature' or 'feature_to_image'.

    Returns:
        ndarray: Bounding boxes in the new coordinate system.
    """
    img_height = img_shape[0]
    img_width = img_shape[1]

    fm_height = np.shape(feature_map)[1]
    fm_width = np.shape(feature_map)[2]
    
    width_scale = img_width / fm_width
    height_scale = img_height / fm_height

    if mode == "image_to_feature":
        boxes[:, [0, 2]] /= width_scale
        boxes[:, [1, 3]] /= height_scale
    elif mode == "feature_to_image":
        boxes[:, [0, 2]] *= width_scale
        boxes[:, [1, 3]] *= height_scale
    return boxes

def convert_proposals(proposals):
    """
    Convert bounding box proposals from 'xyxy' to 'yxyx' format.

    Args:
        proposals (ndarray): Bounding box proposals in 'xyxy' format.

    Returns:
        ndarray: Bounding box proposals in 'yxyx' format.
    """
    batch = np.shape(proposals)[0]
    num_proposals = np.shape(proposals)[1]
    converted_proposals = np.zeros(shape=(batch, num_proposals, 4))

    for i, proposals in enumerate(proposals):
        for j, proposal in enumerate(proposals):
            x1, y1, x2, y2 = proposal
            converted_proposal = [y1, x1, y2, x2]
            converted_proposals[i, j, :] = converted_proposal
    return converted_proposals
