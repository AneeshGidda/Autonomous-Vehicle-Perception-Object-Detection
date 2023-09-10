import tensorflow as tf
import numpy as np
from convert import convert_format

# Function to generate proposals based on anchors and offsets
def generate_proposals(anchors, offsets):
    """
    Generate proposals based on anchors and offsets.

    Args:
        anchors (Tensor): Anchors in (x1, y1, x2, y2) format.
        offsets (Tensor): Offsets for adjusting the anchors.

    Returns:
        Tensor: Generated proposals in (x1, y1, x2, y2) format.
    """
    # Convert anchors from (x1, y1, x2, y2) to (cx, cy, width, height)
    anchors = convert_format(anchors, mode="xyxy_to_cxcywh")

    proposals = np.zeros_like(anchors)
    proposals[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    proposals[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    proposals[:, 2] = anchors[:, 2] * tf.exp(offsets[:, 2])
    proposals[:, 3] = anchors[:, 3] * tf.exp(offsets[:, 3])

    # Convert proposals back to (x1, y1, x2, y2) format
    proposals = convert_format(proposals, mode="cxcywh_to_xyxy")
    return np.array(proposals, dtype="float32")

# Function to filter top proposals based on scores
def filter_proposals(proposals, scores, number_of_proposals):
    """
    Filter top proposals based on scores.

    Args:
        proposals (Tensor): Proposals in (x1, y1, x2, y2) format.
        scores (Tensor): Scores associated with each proposal.
        number_of_proposals (int): Number of top proposals to keep.

    Returns:
        Tensor: Filtered proposals in (x1, y1, x2, y2) format.
        Tensor: Filtered scores for the selected proposals.
    """
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
