from floewise_img_process import floewise_img_process
from tqdm import tqdm
from getimages import get_images
import time

# Generates CSV with pixel confusion matrix for each image.
def main():
    # Run image processor

    IFT_RESULTS_PATH = '../data/ift_results'
    VALIDATION_IMG_PATH = '../data/validation_images/labeled_floes_png'

    LAND_MASK_PATH = '../data/validation_images/landmask'

    complete_cases = get_images(IFT_RESULTS_PATH, VALIDATION_IMG_PATH, LAND_MASK_PATH)



    for index, row in tqdm(complete_cases.iterrows()):
        
        fps, fns, ift_to_man = floewise_img_process(row['manual_path'], row['ift_path'], row['start_date'], 
                                    row['satellite'], float(row['dx_km']), str(row['land_mask_path']), 15)
    

if __name__ == '__main__':
    main()