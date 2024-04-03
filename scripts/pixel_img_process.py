import imageio.v3 as iio
from PIL import Image
import numpy as np
import os

def pixel_image_process(
                manual_path: str, 
                ift_path: np.ndarray, 
                case_no: str, 
                satellite: str, 
                land_mask_path: str, 
                algorithm_name: str,
                save_images: bool = False
                ):
  
    
    labeled_image = ift_path

    # Retrieve land mask and dilate if desired.
    land_mask_img = iio.imread(land_mask_path)

    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0

    # Retrieve manual floes from PNG and calculate intersections
    manual_image = iio.imread(manual_path)
    new_img = np.zeros_like(manual_image[:,:,0])

    labeled_image[labeled_image[:,:] > 0] = 255
    labeled_image[land_mask_img[:,:,0] > 0] = 0
    new_img[manual_image[:,:,0] > 0] = 255

    true_pos = np.sum(np.logical_and(labeled_image[:,:] > 0, new_img[:,:] > 0))
    false_pos = np.sum(np.logical_and(labeled_image[:,:] > 0, new_img[:,:] == 0))
    false_neg = np.sum(np.logical_and(labeled_image[:,:] == 0, new_img[:,:] > 0))
    true_neg = np.sum(np.logical_and(labeled_image[:,:] == 0, new_img[:,:] == 0))
    

    if save_images:
        dir_name = f'./out_{algorithm_name}/images'

        if not os.path.exists(f'./out_{algorithm_name}'):
            os.mkdir(f'./out_{algorithm_name}')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not os.path.exists(dir_name + '/manual'):
            os.mkdir(dir_name + '/manual')
        if not os.path.exists(dir_name + '/landmask'):
            os.mkdir(dir_name + '/landmask')
        if not os.path.exists(dir_name + '/ift'):
            os.mkdir(dir_name + '/ift')
        if not os.path.exists(dir_name + '/overlaid'):
            os.mkdir(dir_name + '/overlaid')

        new_img_im = Image.fromarray(new_img.astype('uint8'))
        new_img_im.save(dir_name + "/manual/results_manual_" + str(case_no) + satellite + ".jpg")

        land_img_im = Image.fromarray(land_mask_img.astype('uint8'))
        land_img_im.save(dir_name + "/landmask/results_landmask_" + str(case_no) + satellite + ".jpg")

        labeled_image_im = Image.fromarray(labeled_image)
        labeled_image_im.save(dir_name + "/ift/results_ift_" + str(case_no) + satellite + ".jpg")

        overlaid_im = Image.blend(labeled_image_im, new_img_im, 0.2)
        overlaid_im.save(dir_name + "/overlaid/overlaid_" + str(case_no) + satellite + ".jpg")


    # Compute absolute confusion matrix

    pixel_confusion_mx_absolute = {'t_pos_pix': int(true_pos), 'f_pos_pix': int(false_pos), 
                                    'f_neg_pix': int(false_neg), 't_neg_pix': int(true_neg)}

    return pixel_confusion_mx_absolute