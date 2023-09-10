import cv2

img_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images\training\image_2\000008.png"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
label_path = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels\training\label_2\000008.txt"

boxes = []
with open(label_path, 'r') as file:
    for line in file:
        data = line.split()
        boxes.append([int(float(data[4])), int(float(data[5])), int(float(data[6])), int(float(data[7]))])

for box in boxes:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()