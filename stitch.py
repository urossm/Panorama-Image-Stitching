import time
import sys

from consts import *
from stitch_utils import *
from utils import *


def stitch_images(images_input, face_blur=True, hdr=True, blend_type=BLEND_TYPES[1]):
    print("========================= STITCHING STARTED ==========================")

    start_timestamp = time.time()

    work_megapix = 1  # Default value = 0.6
    seam_megapix = 0.1  # Default value = 0.1
    compose_megapix = 0  # Default value = 0
    conf_thresh = 1.0  # Default value = 1.0
    ba_refine_mask = 'xxxxx'  # Default value = 'xxxxx'
    warp_type = 'spherical'  # Default value = 'spherical'
    matcher_type = 'homography'  # Default value = 'homography'
    match_conf = 0.65  # Default value = 0.65
    range_width = -1  # Default value = -1
    expos_comp_nr_feeds = 1  # Default value = 1
    expos_comp_block_size = 32  # Default value = 32

    finder = FEATURES_FIND_CHOICES['orb']()
    wave_correct = WAVE_CORRECT_CHOICES['horiz']
    estimator = ESTIMATOR_CHOICES['homography']()
    adjuster = BA_COST_CHOICES['ray']()
    expos_comp_type = EXPOS_COMP_CHOICES['gain_blocks']
    seam_finder = SEAM_FIND_CHOICES['gc_color']

    seam_work_aspect = 1
    work_scale = 1
    seam_scale = 1

    full_img_sizes = []
    features = []
    images = []

    is_work_scale_set = False
    is_seam_scale_set = False

    # ------------------------------------------------------------------------------------------------------------------------

    for i, image in enumerate(images_input):
        full_img = image
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            if seam_megapix > 0:
                seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            else:
                seam_scale = 1.0
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        img_feat = cv.detail.computeImageFeatures2(finder, img)
        features.append(img_feat)
        img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        sys.stdout.write('\r' + "Processing features: " + "." * int(i/len(images_input)*10))
        images.append(img)

    if matcher_type == 'orb':
        match_conf = 0.3
    if range_width == -1:
        matcher = cv.detail_BestOf2NearestMatcher(True, match_conf)
    else:
        matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, True, match_conf)

    p = matcher.apply2(features)

    features_timestamp = time.time()
    sys.stdout.write('\r' + "Features processing time: " + str(round((features_timestamp - start_timestamp), 1)) + " seconds")
    print("")
    sys.stdout.write('\r' + "Adjusting cameras...")

    num_images = len(images)
    if num_images < 2:
        print("Need more images")
        return None

    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        return None
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster.setConfThresh(conf_thresh)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        return None
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    if wave_correct is not None:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
        rmats = cv.detail.waveCorrect(rmats, wave_correct)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]

    cameras = correct_cameras(cameras)

    cameras_timestamp = time.time()

    sys.stdout.write('\r' + "Cameras adjusting time: " + str(round((cameras_timestamp - features_timestamp), 1)) + " seconds")
    print("")

    # ------------------------------------------------------------------------------------------------------------------------

    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []

    if blend_type is BLEND_TYPES[0]:
        sys.stdout.write('\r' + "Finding seams...")
        warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)
        for idx in range(0, num_images):
            select_index = 0
            um = cv.UMat(255 * np.ones((images[idx].shape[0], images[idx].shape[1]), np.uint8))
            masks.append(um)
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REPLICATE)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(masks[select_index], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())
            select_index += 1

        images_warped_float = []
        for img in images_warped:
            img_float = img.astype(np.float32)
            images_warped_float.append(img_float)

        # expos_comp_nr_filtering = 2
        if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
            compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
            # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
        elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
            compensator = cv.detail_BlocksChannelsCompensator(expos_comp_block_size, expos_comp_block_size, expos_comp_nr_feeds)
            # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
        else:
            compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)

        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        masks_warped = seam_finder.find(images_warped_float, corners, masks_warped)

        seam_finding_timestamp = time.time()
        sys.stdout.write('\r' + "Seam finding time: " + str(round((seam_finding_timestamp - cameras_timestamp), 1)) + " seconds")
        print("")

    # ------------------------------------------------------------------------------------------------------------------------

    compose_scale = 1
    corners = []
    masks = []
    warped_images = []
    sizes = []
    select_index = 0

    if compose_megapix > 0:
        compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (images_input[0].shape[0] * images_input[0].shape[1])))
    compose_work_aspect = compose_scale / work_scale
    warped_image_scale *= compose_work_aspect
    warper = cv.PyRotationWarper(warp_type, warped_image_scale)
    for i in range(0, len(images_input)):
        cameras[i].focal *= compose_work_aspect
        cameras[i].ppx *= compose_work_aspect
        cameras[i].ppy *= compose_work_aspect
        sz = (int(round(full_img_sizes[i][0] * compose_scale)), int(round(full_img_sizes[i][1] * compose_scale)))
        K = cameras[i].K().astype(np.float32)
        roi = warper.warpRoi(sz, K, cameras[i].R)
        corners.append(roi[0:2])
        sizes.append(roi[2:4])

    for idx, name in enumerate(images_input):
        sys.stdout.write('\r' + "Warping images: " + str(round((idx / len(images_input) * 100), 1)) + "%")
        img = images_input[idx]
        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(src=img, dsize=None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
        _img_size = (img.shape[1], img.shape[0])

        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REPLICATE)

        if blend_type is BLEND_TYPES[0]:
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            compensator.apply(select_index, corners[idx], image_warped, mask_warped)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
            dilated_mask = cv.dilate(masks_warped[select_index], kernel)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            masks.append(mask_warped)
        warped_images.append(image_warped)
        select_index += 1

    corners_temp = []

    for idx in range(len(images_input)):
        corners_temp.append(corners[idx])

    corners = corners_temp

    # ------------------------------------------------------------------------------------------------------------------------

    full_size_warping_timestamp = time.time()

    if blend_type is BLEND_TYPES[0]:
        sys.stdout.write('\r' + "Images warping time: " + str(round((full_size_warping_timestamp - seam_finding_timestamp), 1)) + " seconds")
        result = blend_seam(warped_images, masks, corners)
    elif blend_type is BLEND_TYPES[1]:
        sys.stdout.write('\r' + "Images warping time: " + str(round((full_size_warping_timestamp - cameras_timestamp), 1)) + " seconds")
        sorted_corners, sorted_images = sort_for_blend(corners, warped_images)
        result = blend_linear(sorted_images, sorted_corners)

    blending_timestamp = time.time()
    sys.stdout.write('\r' + "Images blending time: " + str(round((blending_timestamp - full_size_warping_timestamp), 1)) + " seconds")

    if face_blur is True:
        result = blur_faces(result)

    if hdr is True:
        result = create_HDR(result)

    postprocessing_timestamp = time.time()
    sys.stdout.write('\n' + "Post processing time: " + str(round((postprocessing_timestamp- blending_timestamp), 1)) + " seconds")

    print('\n' + "======================================================================")

    return result
