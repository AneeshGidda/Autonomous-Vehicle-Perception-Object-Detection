import numpy as np

def create_roi_grid(region_of_interest, pooling_height, pooling_width):
    """
    Create a grid of regions of interest (ROIs) for ROIAlign.

    Args:
        region_of_interest (list): A list [x_min, y_min, x_max, y_max] defining the ROI.
        pooling_height (int): Height of the ROI grid.
        pooling_width (int): Width of the ROI grid.

    Returns:
        np.ndarray: An array representing the ROI grid.
    """
    x_min, y_min, x_max, y_max = region_of_interest
    grid_width = x_max - x_min
    grid_height = y_max - y_min
    
    # Calculate the width and height of each square in the ROI grid
    square_width = grid_width / pooling_width
    square_height = grid_height / pooling_height
    
    # Initialize an array to store the ROI grid
    roi_grid = np.zeros(shape=(pooling_height, pooling_width, 4))
    
    # Loop through each row and column of the grid
    for row in range(int(pooling_height)):
        for col in range(int(pooling_width)):
            # Calculate the coordinates of the current square
            square_x_min = x_min + col * square_width
            square_y_min = y_min + row * square_height
            square_x_max = square_x_min + square_width
            square_y_max = square_y_min + square_height
    
            # Create a square defined by [x_min, y_min, x_max, y_max]
            square = np.array([square_x_min, square_y_min, square_x_max, square_y_max])
    
            # Store the square in the ROI grid
            roi_grid[row, col] = square
    return roi_grid

def generate_sample_points(region_of_interest, horizontal_sampling_points=2, vertical_sampling_points=2):
    """
    Generate sampling points within an ROI.

    Args:
        region_of_interest (list): A list [x_min, y_min, x_max, y_max] defining the ROI.
        horizontal_sampling_points (int): Number of horizontal sampling points.
        vertical_sampling_points (int): Number of vertical sampling points.

    Returns:
        np.ndarray: An array representing the sampling points.
    """
    x_min, y_min, x_max, y_max = region_of_interest
    width = x_max - x_min
    height = y_max - y_min

    sampling_points = []
    for iy in range(vertical_sampling_points):
        y_coordinate = y_min + (height / (vertical_sampling_points + 1)) * iy
        for ix in range(horizontal_sampling_points):
            x_coordinate = x_min + (width / (horizontal_sampling_points + 1)) * ix
            sampling_points.append([x_coordinate, y_coordinate])
    return np.array(sampling_points)

def bilinear_interpolation(feature_map, sampling_points, image_num):
    """
    Perform bilinear interpolation to sample values from the feature map.

    Args:
        feature_map (np.ndarray): The input feature map.
        sampling_points (np.ndarray): An array of sampling points.
        image_num (int): Index of the input image.

    Returns:
        np.ndarray: An array of interpolated values.
    """
    height = np.shape(feature_map)[1]
    width = np.shape(feature_map)[2]

    number_of_sampling_points = np.shape(sampling_points)[0]
    number_of_channels = np.shape(feature_map)[3]
    output = np.zeros(shape=(number_of_sampling_points, number_of_channels))

    for i, (x, y) in enumerate(sampling_points):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1

        x0 = max(0, min(x0, width - 1))
        x1 = max(0, min(x1, width - 1))
        y0 = max(0, min(y0, height - 1))
        y1 = max(0, min(y1, height - 1))

        f00 = feature_map[image_num, y0, x0, :]
        f01 = feature_map[image_num, y1, x0, :]
        f10 = feature_map[image_num, y0, x1, :]
        f11 = feature_map[image_num, y1, x1, :]

        dx = x - x0
        dy = y - y0
        interpolated_value = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11
        output[i, :] = interpolated_value
    return output

def roi_align(feature_map, all_proposals, pooling_height, pooling_width):
    """
    Perform ROIAlign operation on a feature map.

    Args:
        feature_map (np.ndarray): The input feature map.
        all_proposals (np.ndarray): An array of ROIs for all images in the batch.
        pooling_height (int): Height of the ROI grid.
        pooling_width (int): Width of the ROI grid.

    Returns:
        np.ndarray: An array of pooled ROI features.
    """
    batch = np.shape(all_proposals)[0]
    for image_num, proposals in enumerate(all_proposals):
        number_of_roi = np.shape(proposals)[0]
        number_of_channels = np.shape(feature_map)[3]
        output = np.zeros((batch, number_of_roi, int(pooling_height), int(pooling_width), number_of_channels))

        for roi_number, region_of_interest in enumerate(proposals):
            roi_grid = create_roi_grid(region_of_interest, int(pooling_height), int(pooling_width))
            for row_number, row in enumerate(roi_grid):
                for col_number, col in enumerate(row):
                    sampling_points = generate_sample_points(col)
                    number_of_sampling_points = np.shape(sampling_points)[0]
                    sampling_point_values = bilinear_interpolation(feature_map, sampling_points, image_num)
                    output[image_num, roi_number, row_number, col_number, :] = np.sum(sampling_point_values, axis=0) / number_of_sampling_points
        return output
