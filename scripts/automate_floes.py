from floewise_img_process import floewise_img_process
from pixel_img_process import pixel_image_process
from tqdm import tqdm
from getimages import get_images
import json

# Generates CSV with pixel confusion matrix for each image.
def process_floes(ift_path, validation_path, land_mask_path, algo_name):
    # Run image processor

    complete_cases = get_images(ift_path, validation_path, land_mask_path)

    results = {}


    for _, row in tqdm(complete_cases.iterrows(), total=len(complete_cases)):


        case = str(row['case_number']) + "_" + row['satellite']

        case_dict = row.to_dict()


        floe_conf_mx, fps, fns, ift_to_man, intersections = floewise_img_process(row['manual_path'], row['ift_path'], 
                                                    row['start_date'], row['satellite'], float(row['dx_km']), 
                                                    str(row['land_mask_path']), 15)

        pix_conf_mx = pixel_image_process(row['manual_path'], row['ift_path'], row['start_date'], 
                                    row['satellite'], float(row['dx_km']), str(row['land_mask_path']), algo_name, 15, save_images=True)

        case_dict.update(pix_conf_mx)
        case_dict.update(floe_conf_mx)
        

        case_dict['fp_floes'] = fps
        case_dict['fn_floes'] = fns
        case_dict['ift_to_man'] = ift_to_man
        case_dict['intersections'] = intersections

        results[case] = case_dict

    with open(f'process_results/out_{algo_name}.json', 'w') as f:
        json.dump(results, f)