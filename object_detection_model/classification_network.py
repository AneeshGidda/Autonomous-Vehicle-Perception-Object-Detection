import tensorflow as tf
from roi_align import roi_align
from loss_functions import calc_classifier_loss

# Define a ClassificationNetwork class that extends tf.keras.Model
class ClassificationNetwork(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()

        # Reshape layer to match the expected input shape
        self.reshape = tf.keras.layers.Reshape(target_shape=(22, -1))
        
        # First fully connected layer with 128 units and ReLU activation
        self.fully_connected_layer1 = tf.keras.layers.Dense(units=128, activation="relu")
        
        # Dropout layer to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        
        # Second fully connected layer with 64 units and ReLU activation
        self.fully_connected_layer2 = tf.keras.layers.Dense(units=64, activation="relu")
        
        # Another dropout layer
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        # Classifier layer with 'num_classes' units and softmax activation
        self.classifier = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def call(self, feature_map, proposals, ground_truth_classes=None):
        # Determine the mode: 'eval' or 'train'
        if ground_truth_classes is None:
            mode = "eval"
        else:
            mode = "train"

        # Perform Region of Interest (ROI) pooling using the provided feature map and proposals
        x = roi_align(feature_map=feature_map, 
                      all_proposals=proposals,
                      pooling_height=4,
                      pooling_width=4)

        # Reshape the pooled features
        x = self.reshape(x)

        # Pass the features through the first fully connected layer and apply ReLU activation
        x = self.fully_connected_layer1(x)
        
        # Apply dropout to the first layer's output
        x = self.dropout1(x)
        
        # Pass the features through the second fully connected layer and apply ReLU activation
        x = self.fully_connected_layer2(x)
        
        # Apply dropout to the second layer's output
        x = self.dropout2(x)
        
        # Generate class probabilities using the classifier layer
        classes = self.classifier(x)

        # In evaluation mode, return the predicted classes
        if mode == "eval":
            return classes
        
        # In training mode, calculate the classifier loss
        if mode == "train":
            classifier_loss = calc_classifier_loss(classes, ground_truth_classes)
            return classifier_loss
