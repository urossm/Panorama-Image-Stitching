import os
import time
import sys

import cv2 as cv
import numpy as np
from collections import OrderedDict
from scipy import stats
from matplotlib import pyplot as plt

EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster

FEATURES_FIND_CHOICES = OrderedDict()
try:
    cv.xfeatures2d_SURF.create()  # check if the function can be called
    FEATURES_FIND_CHOICES['surf'] = cv.xfeatures2d_SURF.create
except (AttributeError, cv.error) as e:
    print("SURF not available")
# if SURF not available, ORB is default
FEATURES_FIND_CHOICES['orb'] = cv.ORB.create
try:
    FEATURES_FIND_CHOICES['sift'] = cv.SIFT_create
except AttributeError:
    print("SIFT not available")
try:
    FEATURES_FIND_CHOICES['brisk'] = cv.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['akaze'] = cv.AKAZE_create
except AttributeError:
    print("AKAZE not available")

SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['gc_color'] = cv.detail_GraphCutSeamFinder('COST_COLOR')
SEAM_FIND_CHOICES['gc_colorgrad'] = cv.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
SEAM_FIND_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
SEAM_FIND_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
SEAM_FIND_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator
ESTIMATOR_CHOICES['affine'] = cv.detail_AffineBasedEstimator

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

WAVE_CORRECT_CHOICES = OrderedDict()
WAVE_CORRECT_CHOICES['horiz'] = cv.detail.WAVE_CORRECT_HORIZ
WAVE_CORRECT_CHOICES['no'] = None
WAVE_CORRECT_CHOICES['vert'] = cv.detail.WAVE_CORRECT_VERT

BLEND_TYPES = (
    'seam',
    'vertical'
)


def combine_two_images(background, foreground, mask, x, y, blend_strength=139):
    image_without_alpha = foreground
    image_with_alpha = cv.cvtColor(foreground, cv.COLOR_BGR2BGRA)
    mask = cv.UMat(mask).get()

    alpha = mask.copy()
    alpha[mask != 0] = 255
    if blend_strength != 0:
        alpha = cv.blur(alpha, (blend_strength, blend_strength))
    image_with_alpha[:, :, 3] = alpha
    foreground = cv.bitwise_and(image_with_alpha, image_with_alpha, mask=alpha)

    height, width, c = foreground.shape

    background_subsection = background[y:y + height, x:x + width]

    alpha_channel = foreground[:, :, 3] / 255
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + image_without_alpha * alpha_mask

    # overwrite the section of the background image that has been updated
    background[y:y + height, x:x + width] = composite

    return background


def generate_mask(width, height, direction):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    if direction == "horizontal":
        midpoint = int(width * 0.4)
        image[:, :midpoint] = (0, 0, 0)
        image[:, midpoint:] = (255, 255, 255)
    elif direction == "vertical":
        midpoint = int(height * 0.35)
        image[:midpoint, :] = (0, 0, 0)
        image[midpoint:, :] = (255, 255, 255)

    image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)

    return image


def blend_seam(images, masks, corners):
    print("")
    # sys.stdout.write('\r' + "Blending images: 0.0%")

    num_images = len(images)

    # Convert the corners from tuples to lists
    corners = [list(corner) for corner in corners]

    # Offsets that represent difference between minimum and maximum corners for each axis
    max_corner_x = max(corners[i][0] for i in range(num_images))
    max_corner_y = max(corners[i][1] for i in range(num_images))
    min_corner_x = min(corners[i][0] for i in range(num_images))
    min_corner_y = min(corners[i][1] for i in range(num_images))
    offset_x = max_corner_x - min_corner_x
    offset_y = max_corner_y - min_corner_y

    # Maximum size for final blended image which is last image size + offset
    canvas_width = max(images[i].shape[1] for i in range(num_images)) + offset_x
    canvas_height = max(images[i].shape[0] for i in range(num_images)) + offset_y

    # Offset corners so it starts from (0,0) and not some random coordinates
    for i in range(num_images):
        corners[i][0] -= min_corner_x
        corners[i][1] -= min_corner_y

    # Create an empty canvas to hold the blended result
    result = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    for i, (image, mask, corner) in enumerate(zip(images, masks, corners)):
        x, y = corner
        if i == 0:
            matrix = np.float32([[1, 0, x], [0, 1, y]])
            image_resized = cv.warpAffine(image, matrix, (canvas_width, canvas_height))
            result = image_resized
        else:
            # print(i, x, y, canvas_width, canvas_height, image.shape[1], image.shape[0])
            result = combine_two_images(result, image, mask, x, y)
            sys.stdout.write('\r' + "Blending images: " + str(round((i / len(images) * 100), 1)) + "%")

    return result


def blend_linear(images, corners, direction="horizontal", blend_strength=139):
    # sys.stdout.write('\r' + "Blending images: 0.0%")

    num_images = len(images)

    corners = [list(corner) for corner in corners]

    max_corner_x = max(corners[i][0] for i in range(num_images))
    max_corner_y = max(corners[i][1] for i in range(num_images))
    min_corner_x = min(corners[i][0] for i in range(num_images))
    min_corner_y = min(corners[i][1] for i in range(num_images))
    offset_x = max_corner_x - min_corner_x
    offset_y = max_corner_y - min_corner_y

    canvas_width = max(images[i].shape[1] for i in range(num_images)) + offset_x
    canvas_height = max(images[i].shape[0] for i in range(num_images)) + offset_y

    for i in range(num_images):
        corners[i][0] -= min_corner_x
        corners[i][1] -= min_corner_y

    result = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    for i, (image, corner) in enumerate(zip(images, corners)):
        x, y = corner
        if i == 0:
            matrix = np.float32([[1, 0, x], [0, 1, y]])
            image_resized = cv.warpAffine(image, matrix, (canvas_width, canvas_height))
            result = image_resized
        else:
            mask = generate_mask(image.shape[1], image.shape[0], direction=direction)
            result = combine_two_images(result, image, mask, x, y, blend_strength=blend_strength)

    return result


def sort_for_blend(corners, images):
    enumerated_coordinates = list(enumerate(corners))

    sorted_coordinates = sorted(enumerated_coordinates, key=lambda c: c[1][0])

    sorted_corners = [c[1] for c in sorted_coordinates]
    original_positions = [c[0] for c in sorted_coordinates]

    sorted_images = []

    for position in original_positions:
        sorted_images.append(images[position])

    # print(sorted_corners)
    # print(original_positions)

    return sorted_corners, sorted_images


def find_outliers(coordinates, threshold=2):
    x_values = [coord[0] for coord in coordinates]
    y_values = [coord[1] for coord in coordinates]

    z_scores_x = np.abs(stats.zscore(x_values))
    z_scores_y = np.abs(stats.zscore(y_values))

    # print("X scores: " + str(z_scores_x))
    # print("Y scores: " + str(z_scores_y))

    outlier_indices = np.where((z_scores_x > threshold) | (z_scores_y > threshold))[0]
    outliers = [coordinates[i] for i in outlier_indices]

    # print(outlier_indices)

    return outliers, outlier_indices, z_scores_x, z_scores_y


def correct_cameras(cameras):
    cameras_params = []
    for i, camera in enumerate(cameras):
        camera_temp = camera
        params = {
            'ID': i,
            'R1': camera.R[0][0],
            'R2': camera.R[0][1],
            'R3': camera.R[0][2],
            'R4': camera.R[1][0],
            'R5': camera.R[1][1],
            'R6': camera.R[1][2],
            'R7': camera.R[2][0],
            'R8': camera.R[2][1],
            'R9': camera.R[2][2],
            # 'aspect': camera.aspect,
            'focal': camera.focal,
            'ppx': camera.ppx,
            'ppy': camera.ppy,
            # 't': camera.t,
        }
        cameras[i].focal = camera_temp.focal
        cameras[i].R = camera_temp.R
        cameras_params.append(params)

    # directory = "test"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # np.savetxt("test/cameras.txt", cameras_params, fmt='%s')

    return cameras


def crop_image_roi(input_image, crop_data):
    corners, warped_images = crop_data
    height, width, channels = input_image.shape

    corners_top = []
    corners_bottom = []

    for i, corner in enumerate(corners):
        corners_top.append(corner[1])
        corners_bottom.append(corner[1] + warped_images[i].shape[0])

    crop_x_start = int(warped_images[0].shape[1] / 2)
    crop_x_end = input_image.shape[1] - int(warped_images[-1].shape[1] / 2)

    crop_y_start = max(corner for corner in corners_top)
    crop_y_end = min(corner for corner in corners_bottom)

    print(corners_top)
    print(corners_bottom)
    print(crop_y_start)
    print(crop_y_end)

    cropped_image = input_image[crop_y_start:crop_y_end, crop_x_start: crop_x_end]

    return cropped_image
