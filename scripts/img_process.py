import imageio.v3 as iio
import h5py
from PIL import Image
import numpy as np
import os
import cv2

def image_process(
                manual_path: str, 
                ift_path: str, 
                date: str, 
                satellite: str, 
                image_width_km: float,
                land_mask_path: str, 
                land_dilation_distance_km: float = 0, 
                save_images: bool = False
                ):
  
    # Retrieve IFT floes from hdf5 file
    with h5py.File(ift_path, "r") as ift_image:
        properties = ift_image['floe_properties']

        labeled_image = properties['labeled_image'][:].astype('uint8')

    img_size = labeled_image.shape

    # Retrieve land mask and dilate if desired.
    land_mask_img = iio.imread(land_mask_path)

    if land_dilation_distance_km != 0:

        pixel_distance = int(round(land_dilation_distance_km * img_size[0] / image_width_km))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel_distance, pixel_distance))

        land_mask_img = cv2.dilate(land_mask_img, kernel)


    labeled_image = np.flipud(labeled_image)
    labeled_image = np.rot90(labeled_image, 3)
 
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0

    # Retrieve manual floes from PNG and calculate intersections
    manual_image = iio.imread(manual_path)

    new_img = np.zeros((len(manual_image), len(manual_image[0])))

    # Loop through pixels in the manually masked image
    for i in range(len(manual_image)):
            for j in range(len(manual_image[i])):
                # Only consider pixels where the landmask is not present when calculating t/f p/n rates.
                if land_mask_img[i][j][0] == 0:
                    if labeled_image[i][j] != 0 and manual_image[i][j][0] != 0:
                        true_pos += 1
                        labeled_image[i][j] = 255
                        new_img[i][j] = 255
                    elif labeled_image[i][j] == 0 and manual_image[i][j][0] != 0:
                        new_img[i][j] = 255
                        false_neg += 1
                    elif labeled_image[i][j] != 0 and manual_image[i][j][0] == 0:
                        labeled_image[i][j] = 255
                        false_pos += 1
                    else:
                        true_neg += 1

                # For the purposes of output image, black out all landmasked pixels
                else:
                    if labeled_image[i][j] != 0 and manual_image[i][j][0] != 0:
                        labeled_image[i][j] = 0
                        new_img[i][j] = 0
                    elif labeled_image[i][j] == 0 and manual_image[i][j][0] != 0:
                        new_img[i][j] = 0
                    elif labeled_image[i][j] != 0 and manual_image[i][j][0] == 0:
                        labeled_image[i][j] = 0
    
    
    if save_images:
        if not os.path.exists('./out_images'):
            os.mkdir('./out_images')
        if not os.path.exists('./out_images/manual'):
            os.mkdir('./out_images/manual')
        if not os.path.exists('./out_images/landmask'):
            os.mkdir('./out_images/landmask')
        if not os.path.exists('./out_images/ift'):
            os.mkdir('./out_images/ift')
        if not os.path.exists('./out_images/overlaid'):
            os.mkdir('./out_images/overlaid')

        new_img_im = Image.fromarray(new_img.astype('uint8'))
        new_img_im.save("./out_images/manual/results_manual_" + date + satellite + ".jpg")

        land_img_im = Image.fromarray(land_mask_img.astype('uint8'))
        land_img_im.save("./out_images/landmask/results_landmask_" + date + satellite + ".jpg")

        labeled_image_im = Image.fromarray(labeled_image)
        labeled_image_im.save("./out_images/ift/results_ift_" + date + satellite + ".jpg")

        overlaid_im = Image.blend(labeled_image_im, new_img_im, 0.2)
        overlaid_im.save("./out_images/overlaid/overlaid_" + date + satellite + ".jpg")


    # Compute absolute confusion matrix

    pixel_confusion_mx_absolute = {'t_pos': true_pos, 'f_pos': false_pos, 'f_neg': false_neg, 't_neg': true_neg}



    return pixel_confusion_mx_absolute