import math
from datetime import datetime
import os

import cv2 as cv
import numpy as np

from consts import *
import time
import string
import random


# ------------------- GENERAL UTILS ------------------------------


def generate_filename(length, key):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string + "-" + key


def get_formatted_date():
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y")
    return formatted_date


# ------------------- IMAGE UTILS ------------------------------


def load_images():
    try:
        images = []

        for filename in os.listdir(IMAGES_FOLDER_NAME):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = cv.imread(os.path.join(IMAGES_FOLDER_NAME, filename))
                if img is not None:
                    images.append(img)

        print("Number of images: " + str(len(images)))
        return images
    except Exception as e:
        print("Image decode failed: ", e)
        return []


def blur_faces(image):
    face_cascade = cv.CascadeClassifier("libs/haarcascades/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image, 1.25, 4)
    for (x, y, w, h) in faces:
        face_roi = image[y:y + h, x:x + w]
        face_roi = cv.GaussianBlur(face_roi, (0, 0), 9)
        image[y:y + h, x:x + w] = face_roi
    return image


def create_HDR(image):
    img_list = []
    exposure_times = np.array([1, 0.4, 0.0333], dtype=np.float32)

    for idx, adjustment in enumerate(exposure_times):
        exposed_image = expose(image, adjustment)
        img_list.append(exposed_image)

    merge_mertens = cv.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    # = cv.convertScaleAbs(res_mertens_8bit, alpha=1)
    adjusted_image = hsv_modification(res_mertens_8bit, 1.3, 1)
    return adjusted_image


def expose(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv.LUT(img, gamma_table)


def hsv_modification(image, saturation, value):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * value, 0, 255)
    modified_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return modified_image
