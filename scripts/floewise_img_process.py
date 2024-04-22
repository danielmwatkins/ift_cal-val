import cv2
from copy import deepcopy
import numpy as np
import h5py
import sys
from pandas import Series
from filter_function import ift_filter

IOU_THRESHOLD = 0.5

def floewise_img_process(
                manual_path: str, 
                ift_path: str, 
                land_mask_path: str, 
                tc_image_path: str,
                fc_image_path: str,
                threshold_params: dict = None
                ):

    
    
    if ift_path.endswith('.h5'):
        h5 = True
        # Retrieve IFT floes from hdf5 file
        with h5py.File(ift_path, "r") as ift_image:
            properties = ift_image['floe_properties']
            
            labeled_image = properties['labeled_image'][:].astype('uint8')

            labeled_image = np.flipud(labeled_image)
            labeled_image = np.rot90(labeled_image, 3)

    elif ift_path.endswith('.tif') or ift_path.endswith('.tiff'):
        labeled_image = cv2.imread(ift_path)
        
        h5 = False
    else:
        print('Invalid image type for IFT predicted floes. Must be .tiff or .h5')
        sys.exit(1)

    tc_img = cv2.imread(tc_image_path)
    fc_img = cv2.imread(fc_image_path)

    props = None

    if threshold_params:
        labeled_image, props = ift_filter(labeled_image, tc_img, fc_img, **threshold_params)
        props = props[~props['flagged']]

    # Retrieve land mask and dilate if desired.
    land_mask_img = cv2.imread(land_mask_path)

    idx_landmass = land_mask_img[:,:,0] > 0

    # Get individual ift floes
    ift_num_labels, ift_labels, ift_stats, ift_centroids = cv2.connectedComponentsWithStats(
                                                                labeled_image, connectivity=8)

    # Create map between original IFT labels and OpenCV results
    # Should be removed when OpenCV removed as dependency
    if threshold_params:
        flat_opencv = Series(ift_labels.flatten()).unique()
        flat_labeled = Series(labeled_image.flatten()).unique()
        open_to_labeled_map = {opencv: labeled for (opencv, labeled) in zip(flat_opencv, flat_labeled)}
        del open_to_labeled_map[0]


    # Manual image loading
    raw_manual_img = cv2.imread(manual_path)
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

        # Check if this floe intersects the landmask
        landmask_intersection = np.logical_and(idx, idx_landmass)

        # If it does, skip this floe for the purposes of calculation
        if np.sum(landmask_intersection) > 0:
            continue

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
            ift_to_manual_floes[int(i)] = overlapping_manual

            # Remove false negatives
            for floe in overlapping_manual:
                if floe in false_negatives:
                    false_negatives.remove(floe)


    # Can consider the similarity of the floe pairings in the dictionary
    for key, value in ift_to_manual_floes.items():

        # Create list for all real floes matching this predicted floe.
        new_val = []


        # Generate countours of this floe for boundary iou and dilate
        floe_img_idx = ift_labels[:,:] == key
        binary_img = np.zeros_like(floe_img_idx, dtype=np.uint8)
        binary_img[floe_img_idx] = 255
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(binary_img)
        cv2.drawContours(contour_img, contours, 0, (255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated_predicted_boundary = cv2.dilate(contour_img, kernel, iterations=1)

        dilated_pred_boundary_idx = np.logical_and(dilated_predicted_boundary[:,:] > 0, ift_labels[:,:] != key)
        dilated_predicted_boundary = np.zeros_like(dilated_pred_boundary_idx, dtype=np.uint8)
        dilated_predicted_boundary[dilated_pred_boundary_idx] = 255

        # Get dilated area
        dilated_predicted_area = np.sum(dilated_predicted_boundary[:,:] == 255)
        
        
        for real_floe in value:


            intersection_stats = {}

            intersection_idx = np.logical_and(ift_labels[:,:] == key, man_labels[:,:] == real_floe)

            intersection_area = np.sum(intersection_idx)

            iou = intersection_area / (man_stats[real_floe][4] + ift_stats[key][4] - intersection_area)

            centroid_distance_px = np.sqrt((man_centroids[real_floe][0] - ift_centroids[key][0])**2 + 
                                (man_centroids[real_floe][1] - ift_centroids[key][1])**2)


            # Get dilated real floe boundary
            floe_img_idx = man_labels[:,:] == real_floe
            binary_img = np.zeros_like(floe_img_idx, dtype=np.uint8)
            binary_img[floe_img_idx] = 255
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = np.zeros_like(binary_img)
            cv2.drawContours(contour_img, contours, 0, (255))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            dilated_real_boundary = cv2.dilate(contour_img, kernel, iterations=1)


            dilated_real_boundary_idx = np.logical_and(dilated_real_boundary[:,:] > 0, man_labels[:,:] == real_floe)
            dilated_real_boundary = np.zeros_like(dilated_real_boundary_idx, dtype=np.uint8)
            dilated_real_boundary[dilated_real_boundary_idx] = 255

            dilated_real_area = np.sum(dilated_real_boundary[:,:] == 255)


            # Calculate boundary iou
            intersection_idx = np.logical_and(dilated_real_boundary[:,:] == 255, dilated_predicted_boundary[:,:] == 255)
            bound_intersection_area = np.sum(intersection_idx)
            boundary_iou = bound_intersection_area / (dilated_real_area + dilated_predicted_area - bound_intersection_area)

            intersection_stats['area_percent_difference'] = (ift_stats[key][4] - man_stats[real_floe][4]) / man_stats[real_floe][4]
            intersection_stats['real_floe'] = int(real_floe)
            intersection_stats['iou'] = iou
            intersection_stats['centroid_distance'] = centroid_distance_px
            intersection_stats['boundary_iou'] = boundary_iou
            new_val.append(intersection_stats)

        ift_to_manual_floes[key] = new_val

    # Copy intersection stats for later
    intersections = deepcopy(ift_to_manual_floes)


    # Non-max suppression
    # Takes care of undersegmentation errors
    for ift, reals in ift_to_manual_floes.items():

        # Gets index of best matching real floe for IFT floe
        possible_match_idx = max(enumerate(reals), key=lambda x: x[1]['iou'])[0]
        
        # Removes best matching from list of IFT floes
        possible_match = reals.pop(possible_match_idx)

        # Adds remainder of floes (besides best match) to false negative list
        for non_match in reals:
            real_floe_number = non_match['real_floe']
            false_negatives.append({'floe_number': real_floe_number, 
                                    'floe_area': int(man_stats[real_floe_number][4]), 
                                    'overlap': True})

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
        false_positives.append({'floe_number': idx, 
                                    'floe_area': int(ift_stats[idx][4]), 'overlap': True})
        del ift_to_manual_floes[idx]

    """
    After NMS, we have a list of FN and FP floes and a dict with predictions as keys and
    best matching real floes as values. Other real floes have been discarded as FNs.
    After NMS, the next step is determining whether the remaining pairings are good pairings.
    """

    to_remove = []
    for ift, real in ift_to_manual_floes.items():
        if real['boundary_iou'] < IOU_THRESHOLD and real['iou'] < IOU_THRESHOLD:
            to_remove.append(ift)

    for idx in to_remove:
        false_positives.append({'floe_number': idx, 
                                'floe_area': int(ift_stats[idx][4]), 'overlap': True})
        real_floe_number = ift_to_manual_floes[idx]['real_floe']
        false_negatives.append({'floe_number': real_floe_number, 
                                'floe_area': int(man_stats[real_floe_number][4]), 'overlap': True})
        del ift_to_manual_floes[idx]

    # Modify false positives to include floe information
    for i in range(len(false_positives)):
        if isinstance(false_positives[i], dict):
            continue
        else:
            floe_number = false_positives[i]
            floe_dict = {}
            floe_dict['floe_number'] = int(floe_number)
            floe_dict['floe_area'] = int(ift_stats[floe_number][4])
            floe_dict['overlap'] = False
            false_positives[i] = floe_dict

        
    # Modify false negatives to include floe information
    for i in range(len(false_negatives)):
        if isinstance(false_negatives[i], dict):
            continue
        else:
            floe_number = false_negatives[i]
            floe_dict = {}
            floe_dict['floe_number'] = int(floe_number)
            floe_dict['floe_area'] = int(man_stats[floe_number][4])
            floe_dict['overlap'] = False
            false_negatives[i] = floe_dict

    floe_conf_matrix = {'t_pos_floes': len(ift_to_manual_floes), 'f_pos_floes': len(false_positives), 
                        'f_neg_floes': len(false_negatives), 't_neg_floes': 1}


    ift_to_manual_tp = {}
    for k, v in ift_to_manual_floes.items():
        ift_to_manual_tp[k] = {'real_floe': v['real_floe'], 'real_floe_area': int(man_stats[v['real_floe']][4]),
                                'ift_floe_area': int(ift_stats[k][4])}

    if threshold_params:
        props['TP'] = ''

        for open, labeled in open_to_labeled_map.items():
            props.loc[props['label'] == labeled, 'TP'] = (open in ift_to_manual_tp.keys())

    return floe_conf_matrix, false_positives, false_negatives, ift_to_manual_tp, intersections, labeled_image, props