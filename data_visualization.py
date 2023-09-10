# Import the OpenCV library
import cv2

# Define the path to the image file
img_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images\training\image_2\000008.png"

# Read the image from the specified path using OpenCV
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# Define the path to the label file corresponding to the image
label_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels\training\label_2\000008.txt"

# Create an empty list to store bounding box coordinates
boxes = []

# Open and read the label file line by line
with open(label_path, 'r') as file:
    for line in file:
        data = line.split()
        # Extract and convert the bounding box coordinates from the label file
        boxes.append([int(float(data[4])), int(float(data[5])), int(float(data[6])), int(float(data[7]))])

# Iterate through the bounding boxes and draw rectangles on the image
for box in boxes:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

# Display the image with bounding boxes using OpenCV
cv2.imshow("img", img)

# Wait for a key press and close the window when a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
