from floewise_img_process import floewise_img_process
from pixel_img_process import pixel_image_process
from tqdm import tqdm
import pandas as pd
from getimages import get_images
import json
    

# Generates CSV with pixel confusion matrix for each image.
def process_floes(ift_path, validation_path, land_mask_path, algo_name, threshold_params: dict = None, suppress_file_outputs: bool = True):
    # Run image processor

    complete_cases = get_images(ift_path, validation_path, land_mask_path)

    results = {}

    data = pd.DataFrame()

    for _, row in tqdm(complete_cases.iterrows(), total=len(complete_cases)):


        case = str(row['case_number']) + "_" + row['satellite']

        case_dict = row.to_dict()


        floe_conf_mx, fps, fns, ift_to_man, intersections, labeled_image, props = floewise_img_process(row['manual_path'], row['ift_path'],  
                                                            row['land_mask_path'], row['tc_path'], row['fc_path'], threshold_params=threshold_params)

        pix_conf_mx = pixel_image_process(row['manual_path'], labeled_image, row['case_number'], 
                                    row['satellite'], str(row['land_mask_path']), algo_name, save_images=not suppress_file_outputs)

        data = pd.concat([data, props])
        

        case_dict.update(pix_conf_mx)
        case_dict.update(floe_conf_mx)
        

        case_dict['fp_floes'] = fps
        case_dict['fn_floes'] = fns
        case_dict['ift_to_man'] = ift_to_man
        case_dict['intersections'] = intersections

        results[case] = case_dict

    with open(f'process_results/out_{algo_name}.json', 'w') as f:
        json.dump(results, f)

    # data.to_csv('df_with_tp_classifications.csv')

    return results