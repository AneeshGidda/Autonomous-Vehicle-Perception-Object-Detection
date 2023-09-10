import numpy as np

def create_roi_grid(region_of_interest, pooling_height, pooling_width):
    x_min, y_min, x_max, y_max = region_of_interest
    grid_width = x_max - x_min
    grid_height = y_max - y_min

    square_width = grid_width / pooling_width
    square_height = grid_height / pooling_height
    roi_grid = np.zeros(shape=(pooling_height, pooling_width, 4))

    for row in range(int(pooling_height)):
        for col in range(int(pooling_width)):
            square_x_min = x_min + col * square_width
            square_y_min = y_min + row * square_height
            square_x_max = square_x_min + square_width
            square_y_max = square_y_min + square_height
            square = np.array([square_x_min, square_y_min, square_x_max, square_y_max])
            roi_grid[row, col] = square
    return roi_grid

# def generate_sample_points(region_of_interest, pooling_height, pooling_width):
#     x_min, y_min, x_max, y_max = region_of_interest
#     width = x_max - x_min
#     height = y_max - y_min


#     grid_height = height / pooling_height
#     grid_width = width / pooling_width
#     rounded_grid_height = np.ceil(height / pooling_height)
#     rounded_grid_width = np.ceil(width / pooling_width)
#     sampling_points = []

#     for iy in range(int(rounded_grid_height)):
#         y_coordinate = y_min + rounded_grid_height + (iy + 0.5) * rounded_grid_height / grid_height
#         for ix in range(int(rounded_grid_width)):
#             x_coordinate = x_min + rounded_grid_width + (ix + 0.5) * rounded_grid_width / grid_width
#             sampling_points.append([x_coordinate, y_coordinate])
#     return np.array(sampling_points)

def generate_sample_points(region_of_interest, horizontal_sampling_points=2, vertical_sampling_points=2):
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

# def bilinear_interpolation(feature_map, sampling_points):
#     height = np.shape(feature_map)[1]
#     width = np.shape(feature_map)[2]

#     number_of_sampling_points = np.shape(sampling_points)[0]
#     number_of_channels = np.shape(feature_map)[3]
#     output = np.zeros(shape=(number_of_sampling_points, number_of_channels))

#     for i, (x, y) in enumerate(sampling_points):
#         y_low = int(y); x_low = int(x)
#         if (y_low >= height - 1):
#             y_high = y_low = height - 1
#             y = y_low
#         else:
#             y_high = y_low + 1

#         if (x_low >= width-1):
#             x_high = x_low = width-1
#             x = x_low
#         else:
#             x_high = x_low + 1

#         ly = y - y_low; lx = x - x_low
#         hy = 1 - ly; hx = 1 - lx
#         w1 = hy * hx; w2 = hy * lx; w3 = ly * hx; w4 = ly * lx
#         interpolated_value = w1 * feature_map[:, y_low, x_low, :] + w2 * feature_map[:, y_low, x_high, :] + \
#                              w3 * feature_map[:, y_high, x_low, :] +  w4 * feature_map[:, y_high, x_high, :]
#         output[i, :] = interpolated_value
#     return output

def bilinear_interpolation(feature_map, sampling_points, image_num):
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

# feature_map = np.random.rand(1, 94, 311, 2048)
# proposals = np.random.rand(1, 10, 4)
# pooling_height = 7
# pooling_width = 7

# roi_grid = create_roi_grid([0, 0, 7, 7], pooling_height, pooling_width)
# print(roi_grid)
# for row_number, row in enumerate(roi_grid):
#     for col_number, col in enumerate(row):
#         sampling_points = generate_sample_points(col, pooling_height, pooling_width)
#         print(np.shape(sampling_points))

# output = roi_align(feature_map, proposals, pooling_height, pooling_width)
# print(np.shape(output))