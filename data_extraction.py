# Import the shutil module, which provides functions for file operations.
import shutil

# Define a function named 'unzip_file' that takes two arguments:
# - zip_file_path: The path to the ZIP file that needs to be extracted.
# - destination_directory: The directory where the contents of the ZIP file will be extracted.
def unzip_file(zip_file_path, destination_directory):
    # Use the 'shutil.unpack_archive' function to extract the contents of the ZIP file.
    # - 'zip_file_path': Path to the ZIP file to be extracted.
    # - 'destination_directory': Directory where the contents will be extracted.
    # This function automatically detects the archive format (e.g., ZIP, TAR) and extracts it accordingly.
    shutil.unpack_archive(zip_file_path, destination_directory)

# Define the path to the first ZIP file you want to extract (image data).
zip_file_path = r"C:\Users\Aneesh\Downloads\data_object_image_2.zip"

# Define the destination directory where the image data will be extracted.
destination_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images"

# Call the 'unzip_file' function with the image data ZIP file path and destination directory as arguments.
# This will extract the image data from the ZIP file to the specified directory.
unzip_file(zip_file_path, destination_directory)

# Define the path to the second ZIP file you want to extract (label data).
zip_file_path = r"C:\Users\Aneesh\Downloads\data_object_label_2.zip"

# Define the destination directory where the label data will be extracted.
destination_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels"

# Call the 'unzip_file' function again with the label data ZIP file path and destination directory as arguments.
# This will extract the label data from the ZIP file to the specified directory.
unzip_file(zip_file_path, destination_directory)
