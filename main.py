import cv2
import time
from utils import *
from stitch import *


def main():
    total_st = time.time()

    images = load_images()
    if len(images) == 0:
        print(IMAGES_LOADING_ERROR)
        return None

    panorama = stitch_images(images, face_blur=True, hdr=True)
    if panorama is None:
        print(STITCHING_ERROR)
        return None

    cv2.imwrite(PANORAMA_IMAGE_NAME, panorama)

    total_et = time.time()
    print("Total stitching time: " + str(round(total_et - total_st, 1)) + " seconds")


if __name__ == "__main__":
    main()
