import tensorflow as tf
from roi_align import roi_align
from loss_functions import calc_classifier_loss

class ClassificationNetwork(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.reshape = tf.keras.layers.Reshape(target_shape=(22, -1))
        self.fully_connected_layer1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.fully_connected_layer2 = tf.keras.layers.Dense(units=64, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def call(self, feature_map, proposals, ground_truth_classes=None):
        if ground_truth_classes is None:
            mode = "eval"
        else:
            mode = "train"

        x = roi_align(feature_map=feature_map, 
                      all_proposals=proposals,
                      pooling_height=4,
                      pooling_width=4)

        x = self.reshape(x)
        x = self.fully_connected_layer1(x)
        x = self.dropout1(x)
        x = self.fully_connected_layer2(x)
        x = self.dropout2(x)
        classes = self.classifier(x)

        if mode == "eval":
            return classes
        
        if mode == "train":
            classifier_loss = calc_classifier_loss(classes, ground_truth_classes)
            return classifier_loss
















    
# # Sample feature map of shape [1, 94, 311, 2048]
# sample_feature_map = np.random.randn(1, 94, 311, 2048).astype(np.float32)

# # Sample proposals (format: [x1, y1, x2, y2])
# sample_proposals = np.array([
#     [100, 100, 200, 200],
#     [150, 150, 250, 250],
#     [200, 200, 300, 300],
#     # ... add more sample proposals
# ])

# # Create an instance of the DetectorNetwork model
# num_classes = 3  # Change this to the number of classes in your problem
# model = DetectorNetwork(num_classes)

# # Convert the sample feature map and proposals to TensorFlow tensors
# tf_feature_map = tf.convert_to_tensor(sample_feature_map)
# tf_proposals = tf.convert_to_tensor(sample_proposals, dtype=tf.float32)

# # Call the model with the sample inputs
# sample_classes = model(tf_feature_map, tf_proposals)

# print("Sample Classes:", sample_classes)
# print("Sample Classes Shape:", sample_classes.shape)