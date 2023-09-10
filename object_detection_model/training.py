import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from object_detection_model import ObjectDetectionModel
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model = ObjectDetectionModel(num_classes=9)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
num_epochs = 2

# img_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images\training\image_2\000008.png"
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img = img.astype("float32") / 255.0
# img = tf.cast(tf.reshape(img, shape=(1, 375, 1242, 3)), dtype="float32")
# label_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels\training\label_2\000008.txt"

# ground_truth_boxes = []
# ground_truth_classes = []

# with open(label_path, 'r') as file:
#     for line in file:
#         data = line.split()
#         ground_truth_boxes.append([data[4] , data[5], data[6], data[7]])
#         ground_truth_classes.append(data[0])

# num_classes = 2
# ground_truth_boxes = np.array(ground_truth_boxes, dtype="float64")
# ground_truth_classes = pd.get_dummies(ground_truth_classes).values
# ground_truth_boxes = np.reshape(ground_truth_boxes, newshape=(1, 10, 4)).astype("float32")
# ground_truth_classes = np.reshape(ground_truth_classes, newshape=(1, 10, 2)).astype("float32")

images = np.load("images.npy")
ground_truth_boxes = np.load("ground_truth_boxes.npy")
ground_truth_classes = np.load("ground_truth_classes.npy")

for epoch in range(3):
    print("\nStart of epoch %d" % (epoch + 1))
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        loss_value = model(images, ground_truth_boxes, ground_truth_classes)
        loss_value = tf.reshape(loss_value, shape=(1,))
        print(f"Loss: {loss_value}")

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)
    trainable_weights = model.trainable_weights
    # print("Trainable_Weights:")
    # for weight in trainable_weights:
        #[len([item for item in grads if item is None]) - len(grads):]
        # print("Weight Name:", weight.name)
        # print("Weight Shape:", weight.shape)
    print("Gradients:")
    print(len(grads))
    print(len([item for item in grads if item is None]))

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
