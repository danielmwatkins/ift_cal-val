import cv2
import numpy as np
import h5py
from PIL import Image
import os
import imageio.v3 as iio

IOU_THRESHOLD = 0.5

def floewise_img_process(
                manual_path: str, 
                ift_path: str, 
                date: str, 
                satellite: str, 
                image_width_km: float,
                land_mask_path: str, 
                land_dilation_distance_km: float = 0, 
                ):
    
        # Retrieve IFT floes from hdf5 file
    with h5py.File(ift_path, "r") as ift_image:
        properties = ift_image['floe_properties']

        labeled_image = properties['labeled_image'][:].astype('uint8')

    img_size = labeled_image.shape

    # Rotate to proper orientation
    labeled_image = np.flipud(labeled_image)
    labeled_image = np.rot90(labeled_image, 3)

    # Retrieve land mask and dilate if desired.
    land_mask_img = iio.imread(land_mask_path)

    if land_dilation_distance_km != 0:

        pixel_distance = int(round(land_dilation_distance_km * img_size[0] / image_width_km))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel_distance, pixel_distance))

        land_mask_img = cv2.dilate(land_mask_img, kernel)

    idx_landmass = land_mask_img[:,:,0] > 0
    labeled_image[idx_landmass] = 0

    # Get rid of gray, only B/W.
    idx_contrast = labeled_image[:,:] > 0
    labeled_image[idx_contrast] = 255

    # Get individual ift floes
    ift_num_labels, ift_labels, ift_stats, ift_centroids = cv2.connectedComponentsWithStats(
                                                                labeled_image, connectivity=8)

    # Manual image loading
    raw_manual_img = iio.imread(manual_path)
    manual_img = np.zeros((len(raw_manual_img), len(raw_manual_img[0])))
    idx = raw_manual_img[:,:,0] > 0
    manual_img[idx] = 255

    # Get individual manual floes
    man_num_labels, man_labels, man_stats, man_centroids = cv2.connectedComponentsWithStats(
                                                            manual_img.astype('uint8'), connectivity=8)

    ift_to_manual_floes = {}
    false_positives = []

    # Assume a floe is a false positive until we find an IFT one that corresponds
    false_negatives = list(range(1, man_num_labels))

    # Get IFT overlaps with real floes
    for i in range(1, ift_num_labels):

        # Get indices of points in ift image corresponding to floes
        idx = ift_labels[:,:] == i

        # Get manual floes in that area.
        manual_area = man_labels[idx]

        # Get the unique numbers of manual floes in that area.
        overlapping_manual = list(set(manual_area))

        # Add to dict based on result
        # If no overlapping real floes, false positive IFT result.
        if len(overlapping_manual) == 1 and overlapping_manual[0] == 0:
            false_positives.append(i)
        
        # Otherwise, add index of real manual floes to dictionary.
        else:
            if 0 in overlapping_manual:
                overlapping_manual.remove(0)
            ift_to_manual_floes[i] = overlapping_manual

            # Remove false negatives
            for floe in overlapping_manual:
                if floe in false_negatives:
                    false_negatives.remove(floe)


    # Can consider the similarity of the floe pairings in the dictionary
    for key, value in ift_to_manual_floes.items():

        # Create list for all real floes matching this predicted floe.
        new_val = []
        
        for real_floe in value:
            intersection_stats = {}

            intersection_idx = np.logical_and(ift_labels[:,:] == key, man_labels[:,:] == real_floe)

            intersection_area = np.sum(intersection_idx)

            iou = intersection_area / (man_stats[real_floe][4] + ift_stats[key][4] - intersection_area)

            centroid_distance_px = np.sqrt((man_centroids[real_floe][0] - ift_centroids[key][0])**2 + 
                                (man_centroids[real_floe][1] - ift_centroids[key][1])**2)

            intersection_stats['real_floe'] = real_floe
            intersection_stats['iou'] = iou
            intersection_stats['centroid_distance'] = centroid_distance_px
            new_val.append(intersection_stats)

        ift_to_manual_floes[key] = new_val


    # Non-max suppression
    # Takes care of undersegmentation errors
    for ift, reals in ift_to_manual_floes.items():

        # Gets index of best matching real floe for IFT floe
        possible_match_idx = max(enumerate(reals), key=lambda x: x[1]['iou'])[0]
        
        # Removes best matching from list of IFT floes
        possible_match = reals.pop(possible_match_idx)

        # Adds remainder of floes (besides best match) to false negative list
        for non_match in reals:
            false_negatives.append(non_match['real_floe'])

        # Sets ift_to_manual key to remaining best match
        ift_to_manual_floes[ift] = possible_match



    # Pair predicted floes to their best predictions
    # Takes care of oversegmentation errors
    to_remove = set()
    for ift1, real1 in ift_to_manual_floes.items():

        for ift2, real2 in ift_to_manual_floes.items():

            if ift1 != ift2 and real1['real_floe'] == real2['real_floe'] and real1['iou'] > real2['iou']:
                to_remove.add(ift2)

    for idx in list(to_remove):
        false_positives.append(idx)
        false_negatives.append(ift_to_manual_floes[idx]['real_floe'])
        del ift_to_manual_floes[idx]

    """
    After NMS, we have a list of FN and FP floes and a dict with predictions as keys and
    best matching real floes as values. Other real floes have been discarded as FNs.
    After NMS, the next step is determining whether the remaining pairings are good pairings.
    """

    to_remove = []
    for ift, real in ift_to_manual_floes.items():
        if real['iou'] < IOU_THRESHOLD:
            to_remove.append(ift)

    for idx in to_remove:
        false_positives.append(idx)
        false_negatives.append(ift_to_manual_floes[idx]['real_floe'])
        del ift_to_manual_floes[idx]

    # Modify false positives to include floe information
    for i in range(len(false_positives)):
        floe_number = false_positives[i]
        floe_dict = {}
        floe_dict['floe_number'] = floe_number
        floe_dict['floe_area'] = ift_stats[floe_number][4]
        false_positives[i] = floe_dict

        
    # Modify false negatives to include floe information
    for i in range(len(false_negatives)):
        floe_number = false_negatives[i]
        floe_dict = {}
        floe_dict['floe_number'] = floe_number
        floe_dict['floe_area'] = man_stats[floe_number][4]
        false_negatives[i] = floe_dict

    return false_positives, false_negatives, ift_to_manual_floes