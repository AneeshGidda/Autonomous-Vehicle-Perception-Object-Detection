import tensorflow as tf
import numpy as np
from convert import convert_format

def generate_proposals(anchors, offsets):
    anchors = convert_format(anchors, mode="xyxy_to_cxcywh")

    proposals = np.zeros_like(anchors)
    proposals[:,0] = anchors[:,0] + offsets[:,0]*anchors[:,2]
    proposals[:,1] = anchors[:,1] + offsets[:,1]*anchors[:,3]
    proposals[:,2] = anchors[:,2] * tf.exp(offsets[:,2])
    proposals[:,3] = anchors[:,3] * tf.exp(offsets[:,3])

    proposals = convert_format(proposals, mode="cxcywh_to_xyxy")
    return np.array(proposals, dtype="float32")

# def filter_proposals(proposals, scores, number_of_proposals=20):
#     print(proposals)
#     print(np.shape(proposals))
#     print(scores)
#     print(np.shape(scores))
#     sorted_indices = np.argsort(scores)
#     print(sorted_indices)
#     print(np.shape(sorted_indices))
#     sorted_indices = sorted_indices[:, ::-1]
#     print(sorted_indices)
#     print(np.shape(sorted_indices))
#     proposals = np.take(proposals, indices=sorted_indices, axis=2)
#     print(proposals)
#     scores = np.take(scores, indices=sorted_indices, axis=1)
#     print(scores)
#     return proposals, scores

def filter_proposals(proposals, scores,  number_of_proposals):
    batch = np.shape(proposals)[0]
    filtered_proposals = np.zeros((batch, number_of_proposals, 4))
    filtered_scores = np.zeros((batch, number_of_proposals))

    for i in range(batch):
        proposal_scores = scores[i]
        sorted_indices = np.argsort(proposal_scores)[::-1]
        top_indices = sorted_indices[:number_of_proposals]
        filtered_proposals[i] = proposals[i, top_indices]
        filtered_scores[i] = proposal_scores[top_indices]
    return filtered_proposals, filtered_scores

# proposals = np.random.rand(2, 3, 4)  # Replace with your actual proposals data
# scores = np.random.rand(2, 3)  # Replace with your actual scores data

# proposals, scores = filter_proposals(proposals, scores)

# print(proposals.shape)
# print(scores.shape)    
