# Import the shutil module, which provides functions for file operations.
import shutil

# Define a function named 'unzip_file' that takes two arguments:
# - zip_file_path: The path to the ZIP file that needs to be extracted.
# - destination_directory: The directory where the contents of the ZIP file will be extracted.
def unzip_file(zip_file_path, destination_directory):
    shutil.unpack_archive(zip_file_path, destination_directory)

# Define paths for the image and label ZIP files.
image_zip_path = r"C:\Users\Aneesh\Downloads\data_object_image_2.zip"
label_zip_path = r"C:\Users\Aneesh\Downloads\data_object_label_2.zip"

# Define destination directories for extracted data.
image_destination = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images"
label_destination = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels"

# Extract image data from the ZIP file.
unzip_file(image_zip_path, image_destination)

# Extract label data from the ZIP file.
unzip_file(label_zip_path, label_destination)
