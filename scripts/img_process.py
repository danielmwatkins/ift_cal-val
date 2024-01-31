import imageio.v3 as iio
import h5py
from PIL import Image
import numpy as np
import random
import os

def image_process(manual_path: str, ift_path: str, date: str, satellite: str, save_images: bool = False):

    # Retrieve IFT floes from hdf5 file
    with h5py.File(ift_path, "r") as ift_image:
        properties = ift_image['floe_properties']

        labeled_image = properties['labeled_image'][:].astype('uint8')

        for i in range(len(labeled_image)):
            for j in range(len(labeled_image[i])):
                if labeled_image[i][j] != 0:
                    labeled_image[i][j] = 255

    labeled_image = np.flipud(labeled_image)
    labeled_image = np.rot90(labeled_image, 3)
 
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0

    # Retrieve manual floes from PNG and calculate intersections
    manual_image = iio.imread(manual_path)

    new_img = np.zeros((len(manual_image), len(manual_image[0])))

    for i in range(len(manual_image)):
            for j in range(len(manual_image[i])):
                if labeled_image[i][j] != 0 and manual_image[i][j][0] != 0:
                    true_pos += 1
                    new_img[i][j] = 255
                elif labeled_image[i][j] == 0 and manual_image[i][j][0] != 0:
                    new_img[i][j] = 255
                    false_neg += 1
                elif labeled_image[i][j] != 0 and manual_image[i][j][0] == 0:
                    false_pos += 1
                else:
                    true_neg += 1

    manual_size = new_img.shape
    
    if save_images:

        if not os.path.exists('./out_images'):
            os.mkdir('./out_images')

        new_img_im = Image.fromarray(new_img.astype('uint8'))
        new_img_im.save("./out_images/results_manual_" + date + satellite + ".jpg")

        labeled_image_im = Image.fromarray(labeled_image)
        labeled_image_im.thumbnail(manual_size)
        labeled_image_im.save("./out_images/results_ift_" + date + satellite + ".jpg")

        overlaid_im = Image.blend(labeled_image_im, new_img_im, 0.2)
        overlaid_im.save("./out_images/overlaid_" + date + satellite + ".jpg")

    # Compute absolute confusion matrix

    pixel_confusion_mx_absolute = {'t_pos': true_pos, 'f_pos': false_pos, 'f_neg': false_neg, 't_neg': true_neg}



    return pixel_confusion_mx_absolute