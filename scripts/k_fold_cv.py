from getimages import get_images
from sklearn.model_selection import KFold
from tqdm import tqdm
from automate_floes import floewise_img_process
from img_analysis import calculate_performance_params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IFT_RESULTS_PATH = '../data/ift_data/ift_pipeline_default/ift_results'
VALIDATION_IMG_PATH = '../data/validation_images/labeled_floes_png'
LAND_MASK_PATH = '../data/validation_images/landmask'

def k_fold_cv(ift, valdiation, land):

    complete_cases = get_images(ift, valdiation, land)

    df_training = complete_cases.sample(frac=2/3, replace=False, random_state=306)
    df_testing = complete_cases.loc[[x for x in complete_cases.index if x not in df_training.index]]

    kf = KFold(n_splits=3, random_state=2024, shuffle=True)

    output_data = []

    for i, (train_index, test_index) in enumerate(kf.split(df_training)):

        param_performance_map = np.zeros((20, 20))
        param_map = np.full((20,20), None, dtype=object)
        training_df = df_training.iloc[train_index]

        for j, solidity in enumerate(tqdm([x/20 for x in range(0, 20)], leave=False)):
            for k, circularity in enumerate(tqdm([x/20 for x in range(0, 20)], leave=False)):


                results = {}

                for _, row in training_df.iterrows():

                    case = str(row['case_number']) + "_" + row['satellite']

                    case_dict = row.to_dict()

                    floe_conf_mx, fps, fns, ift_to_man, intersections, labeled_image, props = floewise_img_process(row['manual_path'], row['ift_path'],  
                                                            row['land_mask_path'], row['tc_path'], row['fc_path'], 
                                                            threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': circularity,
                                                            'solidity_threshold': solidity,
                                                            'tc_intensity_thresholds': (0, 0, 0),
                                                            'fc_intensity_thresholds': (0, 0, 0)})


                    case_dict.update(floe_conf_mx)
                    case_dict['fp_floes'] = fps
                    case_dict['fn_floes'] = fns
                    case_dict['ift_to_man'] = ift_to_man
                    case_dict['intersections'] = intersections
                    results[case] = case_dict

                floe_vals = {"t_pos": 0, "f_pos": 0, "t_neg": 0, "f_neg": 0}
                for _, data in results.items():

                    for key, _ in floe_vals.items():
                        floe_vals[key] += data[key + "_floes"]

                F1 = calculate_performance_params(floe_vals, object_wise=True, beta = 2)['Floe F1']

                param_performance_map[j, k] = F1
                param_map[j, k] = (solidity, circularity)

        extent = [0, 1, 0, 1]

        plt.imshow(np.flipud(param_performance_map), extent=extent, cmap='hot', interpolation='nearest')
        plt.colorbar(label='F_1 Score')
        plt.xlabel('Circularity')
        plt.ylabel('Solidity')
        plt.title(f'Heatmap of F_beta, k={i}')

        plt.savefig(f"{i}.png", dpi=300)
        plt.close()

        best_index = np.argmax(param_performance_map)

        best_index_2d = np.unravel_index(best_index, param_performance_map.shape)

        best_params = param_map[best_index_2d]

        # Testing data
        for _, row in df_training.iloc[test_index].iterrows():

            case = str(row['case_number']) + "_" + row['satellite']

            case_dict = row.to_dict()

            floe_conf_mx, fps, fns, ift_to_man, intersections, _, _ = floewise_img_process(row['manual_path'], row['ift_path'],  
                                                    row['land_mask_path'], row['tc_path'], row['fc_path'], 
                                                    threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': best_params[1],
                                                    'solidity_threshold': best_params[0],
                                                    'tc_intensity_thresholds': (0, 0, 0),
                                                    'fc_intensity_thresholds': (0, 0, 0)})


            case_dict.update(floe_conf_mx)
            case_dict['fp_floes'] = fps
            case_dict['fn_floes'] = fns
            case_dict['ift_to_man'] = ift_to_man
            case_dict['intersections'] = intersections
            results[case] = case_dict

        floe_vals = {"t_pos": 0, "f_pos": 0, "t_neg": 0, "f_neg": 0}
        for _, data in results.items():

            for key, _ in floe_vals.items():
                floe_vals[key] += data[key + "_floes"]

        F1 = calculate_performance_params(floe_vals, object_wise=True, beta = 2)['Floe F1']

        k_output_data = {'k': i, 'Best solidity': best_params[0], 'Best circularity': best_params[1], 'F_Beta Score': F1}

        output_data.append(k_output_data)
        

    output_data = pd.DataFrame(output_data)

    output_data.to_csv('out.csv')


if __name__ == '__main__':
    k_fold_cv(IFT_RESULTS_PATH, VALIDATION_IMG_PATH, LAND_MASK_PATH)
