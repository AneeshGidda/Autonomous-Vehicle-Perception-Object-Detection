import tensorflow as tf
import numpy as np
from scores_and_offsets import create_iou_matrix, calculate_offsets
from convert import convert_format, convert_boxes
from tqdm import tqdm

def compute_anchor_sizes(base_size, scales, ratios):
    anchor_sizes = []

    for scale in scales:
        for ratio in ratios:
            width = np.round(base_size * scale * np.sqrt(ratio))
            height = np.round(base_size * scale / np.sqrt(ratio))
            anchor_sizes.append((width, height))
    return anchor_sizes

def compute_anchor_centers(feature_map):
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]

    anchor_x = tf.range(0, width, dtype="float64") + 0.5
    anchor_y = tf.range(0, height, dtype="float64") + 0.5
    return anchor_x, anchor_y

def generate_anchor_boxes(anchor_x, anchor_y, anchor_sizes):
    anchor_boxes = []
    for y in tqdm(anchor_y, desc="creating anchors"):
        for x in anchor_x:
            for w, h in anchor_sizes:
                xmin = x - w / 2
                ymin = y - h / 2
                xmax = x + w / 2
                ymax = y + h / 2
                anchor_boxes.append([xmin, ymin, xmax, ymax])
    return np.array(anchor_boxes)



































def anchors(feature_map, ground_truth_boxes=None, object_iou_threshold = 0.7):
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]
    
    if ground_truth_boxes is None:
        mode = "eval"
    else:
        mode = "train"

    base_size = 64
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
    valid_map = valid_indices[0]
    valid_map = np.array(valid_map)

    if mode == "eval":
        return anchor_boxes, valid_map

    ground_truth_boxes = convert_boxes(ground_truth_boxes, [375, 1242, 3], feature_map, mode="image_to_feature")
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
    positive_anchors = anchor_boxes[positive_anchors_index]
    image_space_positive_anchors = convert_boxes(positive_anchors, [375, 1242, 3], feature_map, mode="feature_to_image")
    image_space_ground_truth_boxes = convert_boxes(ground_truth_boxes, [375, 1242, 3], feature_map, mode="feature_to_image")

    # img_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images\training\image_2\000008.png"
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # for box in image_space_positive_anchors:
    #     pt1 = (int(box[0]), int(box[1]))
    #     pt2 = (int(box[2]), int(box[3]))
    #     cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    
    # for box in image_space_ground_truth_boxes:
    #     pt1 = (int(box[0]), int(box[1]))
    #     pt2 = (int(box[2]), int(box[3]))
    #     cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

    # cv2.imshow('Anchors', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    corresponding_ground_truth_boxes = image_space_ground_truth_boxes[max_iou_per_anchor_index, :]
    positive_corresponding_ground_truth_boxes = corresponding_ground_truth_boxes[positive_anchors_index]
    regression_targets = calculate_offsets(image_space_positive_anchors, positive_corresponding_ground_truth_boxes)
    return image_space_positive_anchors, positive_anchors_index, regression_targets, valid_map

# Example feature map with dimensions (batch_size, height, width, num_channels)
# feature_map = np.random.rand(1, 50, 50, 2048)
# feature_map = np.random.rand(1, 94, 311, 2048)

# Example ground truth boxes (format: [y_min, x_min, y_max, x_max])
# Two boxes are defined here
# ground_truth_boxes = np.array([
#     [100, 140, 160, 240],  # Box 1
#     [0, 30, 130, 150]   # Box 2
# ], dtype="float64")

# label_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels\training\label_2\000008.txt"
# ground_truth_boxes = []
# ground_truth_classes = []

# with open(label_path, 'r') as file:
#     for line in file:
#         data = line.split()
#         ground_truth_boxes.append([data[4] , data[5], data[6], data[7]])
#         ground_truth_classes.append(data[0])

# ground_truth_boxes = np.array(ground_truth_boxes[0:6], dtype="float64")

# start_time = time.time()
# anchors(feature_map, ground_truth_boxes)
# end_time = time.time()

# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")


    


