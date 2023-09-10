import tensorflow as tf
import numpy as np
import cv2
from anchor_train import anchor_train
from anchor_eval import anchors_eval
from convert import convert_proposals
from loss_functions import calc_offset_loss, calc_score_loss
from proposals import generate_proposals, filter_proposals

class RegionProposalNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.num_of_anchors = 9
        self.conv1 = tf.keras.layers.Conv2D(filters=2048, kernel_size=(3,3), padding="same", activation="relu", trainable=True)
        self.classifier = tf.keras.layers.Conv2D(filters=self.num_of_anchors, kernel_size=(1,1), padding="same", activation="sigmoid", trainable=True)
        self.regressor = tf.keras.layers.Conv2D(filters=4 * self.num_of_anchors, kernel_size=(1,1), padding="same", activation="linear", trainable=True)

    def call(self, feature_map, ground_truth_boxes=None):
        if ground_truth_boxes is None:
            mode = "eval"
        else:
            mode = "train"

        x = self.conv1(feature_map)
        scores = self.classifier(x)
        offsets = self.regressor(x)

        if mode == "eval":
            anchors, pred_scores, pred_offsets = anchors_eval(feature_map, scores, offsets)

        if mode == "train":
            anchors, pred_scores, pred_offsets, real_scores, real_offsets = anchor_train(feature_map, scores, offsets, ground_truth_boxes)
            classifier_loss = calc_score_loss(pred_scores, real_scores)
            offset_loss = calc_offset_loss(pred_offsets, real_offsets)
            total_loss = classifier_loss + offset_loss

        proposals = generate_proposals(anchors, pred_offsets)
        num_ground_truth_boxes = np.shape(ground_truth_boxes)[1]
        num_proposals = max(num_ground_truth_boxes, np.shape(proposals)[1])
        proposals, scores = filter_proposals(proposals, pred_scores, number_of_proposals=num_proposals)
        nms_proposals = convert_proposals(proposals)

        final_proposals = []
        for proposals, scores in zip(nms_proposals, pred_scores):
            indices = tf.image.non_max_suppression(boxes=proposals, max_output_size=num_ground_truth_boxes, scores=scores, iou_threshold=0.7)
            proposals = tf.gather(proposals, indices=indices)
            final_proposals.append(proposals)
        
        if mode == "eval":
            return final_proposals
        
        if mode == "train":
            return final_proposals, total_loss
    
# feature_map = np.random.rand(1, 20, 20, 2048)
# num_boxes = 7
# ground_truth_boxes = np.random.randint(0, 20, size=(num_boxes, 4))
# ground_truth_boxes[:, 2:] += ground_truth_boxes[:, :2]
# ground_truth_boxes = np.reshape(ground_truth_boxes, newshape=(1, num_boxes, 4)).astype("float32")
# print(np.shape(ground_truth_boxes))
# model = RegionProposalNetwork()
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# for epoch in range(10):
#     with tf.GradientTape() as tape:
#         loss = model(feature_map, ground_truth_boxes)

#     # Print the trainable variables and their shapes
#     trainable_variables = model.trainable_variables
#     print("Trainable Variables:")
#     for var in trainable_variables:
#         print("Variable Name:", var.name)
#         print("Variable Shape:", var.shape)

#     # Calculate gradients
#     grads = tape.gradient(loss, trainable_variables)
#     print("Gradients:")
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     avg_loss = tf.reduce_mean(loss)
#     print(f"Epoch [{epoch + 1}/{10}] Loss: {avg_loss.result().numpy():.4f}")






















# if mode == "eval":
#             scores = np.reshape(scores, newshape=(batch, -1))
#             box_deltas = np.reshape(box_deltas, newshape=(batch, -1, 4))
#             final_proposals = []

#             for box_deltas, scores in zip(box_deltas, scores):
#                 anchor_boxes, valid_map = anchors(feature_map)
#                 scores = scores[valid_map]
#                 box_deltas = box_deltas[valid_map]

#                 number_of_anchors = np.shape(anchor_boxes)[0]
#                 objectness_score = np.full(number_of_anchors, 0)
#                 objectness_score[scores >= 0.7] = 1

#                 positive_anchors_index = np.where(objectness_score == 1)
#                 positive_anchors_index = np.squeeze(positive_anchors_index)
#                 positive_anchors = anchor_boxes[positive_anchors_index]
#                 box_deltas = box_deltas[positive_anchors_index, :]

#                 image_space_positive_anchors = convert_boxes(positive_anchors, [375, 1242, 3], feature_map, mode="feature_to_image")
#                 image_space_positive_anchors = tf.cast(image_space_positive_anchors, dtype="float64")
#                 box_deltas = tf.cast(box_deltas, dtype="float64")
#                 proposals = generate_proposals(image_space_positive_anchors, box_deltas)
#                 final_proposals.append(proposals) 
#             proposals = np.squeeze(np.array(final_proposals))
