from img_process import image_process
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from getimages import get_images

# Generates CSV with pixel confusion matrix for each image.
def main():
    # Run image processor

    IFT_RESULTS_PATH = 'data/ift_results'
    VALIDATION_IMG_PATH = 'data/validation_images/labeled_floes_png'
    LAND_MASK_PATH = 'data/validation_images/landmask'

    complete_cases = get_images(IFT_RESULTS_PATH, VALIDATION_IMG_PATH, LAND_MASK_PATH)

    false_neg = []
    false_pos = []
    true_neg = []
    true_pos = []

    for index, row in tqdm(complete_cases.iterrows()):
        pix_conf_mx = image_process(row['manual_path'], row['ift_path'], row['start_date'], 
        row['satellite'], str(row['land_mask_path']), True)
        false_neg.append(pix_conf_mx['f_neg'])
        true_neg.append(pix_conf_mx['t_neg'])
        false_pos.append(pix_conf_mx['f_pos'])
        true_pos.append(pix_conf_mx['t_pos'])

    
    complete_cases.insert(21, "false_positive", false_pos)
    complete_cases.insert(22, "false_negative", false_neg)
    complete_cases.insert(23, "true_positive", true_pos)
    complete_cases.insert(24, "true_negative", true_neg)
    complete_cases.to_csv('out.csv')
    

if __name__ == '__main__':
    main()