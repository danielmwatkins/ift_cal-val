from pixel_img_process import pixel_image_process
from tqdm import tqdm
from getimages import get_images

# Generates CSV with pixel confusion matrix for each image.
def process_pixels(ift_path, validation_path, land_mask_path):

    complete_cases = get_images(ift_path, validation_path, land_mask_path)


    false_neg = []
    false_pos = []
    true_neg = []
    true_pos = []


    for index, row in tqdm(complete_cases.iterrows()):
        pix_conf_mx = pixel_image_process(row['manual_path'], row['ift_path'], row['start_date'], 
                                    row['satellite'], float(row['dx_km']), str(row['land_mask_path']), 15, True)

        false_neg.append(pix_conf_mx['f_neg'])
        true_neg.append(pix_conf_mx['t_neg'])
        false_pos.append(pix_conf_mx['f_pos'])
        true_pos.append(pix_conf_mx['t_pos'])

    
    complete_cases.insert(21, "false_positive", false_pos)
    complete_cases.insert(22, "false_negative", false_neg)
    complete_cases.insert(23, "true_positive", true_pos)
    complete_cases.insert(24, "true_negative", true_neg)
    complete_cases.to_csv('out.csv')

