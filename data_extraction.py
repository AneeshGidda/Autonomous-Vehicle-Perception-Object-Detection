import shutil

def unzip_file(zip_file_path, destination_directory):
    shutil.unpack_archive(zip_file_path, destination_directory)

zip_file_path = r"C:\Users\Aneesh\Downloads\data_object_image_2.zip"
destination_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_images"

unzip_file(zip_file_path, destination_directory)

zip_file_path = r"C:\Users\Aneesh\Downloads\data_object_label_2.zip"
destination_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_labels"

unzip_file(zip_file_path, destination_directory)

zip_file_path = r"c:\Users\Aneesh\Downloads\data_object_velodyne.zip"
destination_directory = r"C:\Users\Aneesh\Codes\Autonomous Vehicle Perception\KITTI_Vision_lidar"

unzip_file(zip_file_path, destination_directory)