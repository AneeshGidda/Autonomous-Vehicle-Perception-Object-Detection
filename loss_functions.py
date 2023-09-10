import tensorflow as tf
import numpy as np

def calc_score_loss(pred_scores, real_scores):
    score_loss = 0
    for real_score, pred_score in zip(real_scores, pred_scores):
        pred_probs = tf.nn.softmax(pred_score)
        cross_entropy_loss = -tf.reduce_sum(real_score * tf.math.log(pred_probs + 1e-10))
        score_loss += cross_entropy_loss
    score_loss = score_loss / len(real_scores)
    return score_loss

def calc_offset_loss(pred_offsets, real_offsets):
    delta_x = pred_offsets[:, 0] - real_offsets[:, 0]
    delta_y = pred_offsets[:, 1] - real_offsets[:, 1]
    delta_w = pred_offsets[:, 2] - real_offsets[:, 2]
    delta_h = pred_offsets[:, 3] - real_offsets[:, 3]

    smooth_l1_x = tf.where(tf.abs(delta_x) < 1, 0.5 * delta_x**2, tf.abs(delta_x) - 0.5)
    smooth_l1_y = tf.where(tf.abs(delta_y) < 1, 0.5 * delta_y**2, tf.abs(delta_y) - 0.5)
    smooth_l1_w = tf.where(tf.abs(delta_w) < 1, 0.5 * delta_w**2, tf.abs(delta_w) - 0.5)
    smooth_l1_h = tf.where(tf.abs(delta_h) < 1, 0.5 * delta_h**2, tf.abs(delta_h) - 0.5)
    offset_loss = tf.reduce_sum(smooth_l1_x + smooth_l1_y + smooth_l1_w + smooth_l1_h)
    return offset_loss

def calc_classifier_loss(all_classes, all_ground_truth_classes):
    classifier_loss = 0
    for classes, ground_truth_classes in zip(all_classes, all_ground_truth_classes):
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_object(ground_truth_classes, classes)
        classifier_loss += loss
    classifier_loss = classifier_loss / len(classes)
    return classifier_loss