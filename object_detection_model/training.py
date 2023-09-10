import tensorflow as tf
import numpy as np
from object_detection_model import ObjectDetectionModel

# Set TensorFlow logging verbosity to ERROR to suppress unnecessary log messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Create an instance of the object detection model with 9 classes
model = ObjectDetectionModel(num_classes=9)

# Define the optimizer (Adam) with a specific learning rate (0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Specify the number of training epochs
num_epochs = 10

# Load preprocessed training data (images, ground truth boxes, ground truth classes)
images = np.load("images.npy")
ground_truth_boxes = np.load("ground_truth_boxes.npy")
ground_truth_classes = np.load("ground_truth_classes.npy")

# Training loop over the specified number of epochs
for epoch in range(num_epochs):
    print("\nStart of epoch %d" % (epoch + 1))
    
    # Use TensorFlow GradientTape to compute gradients for model parameters
    with tf.GradientTape() as tape:
        # Compute the loss for the model using the loaded data
        loss_value = model(images, ground_truth_boxes, ground_truth_classes)
        
        # Reshape the loss for printing (if needed)
        loss_value = tf.reshape(loss_value, shape=(1,))
        
        # Print the current loss value
        print(f"Loss: {loss_value}")

    # Calculate gradients of the loss with respect to model parameters
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Apply gradients to update model parameters using the optimizer
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
