import os
import numpy as np
from img_analysis import analyze_algo
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

IFT_RESULTS_PATH = '../data/ift_results'
VALIDATION_IMG_PATH = '../data/validation_images/labeled_floes_png'
LAND_MASK_PATH = '../data/validation_images/landmask'

def main():


    original_pix, original_floe, original_fsd_plot = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                    validation_path= '../data/validation_images/labeled_floes_png', 
                    land_mask_path= '../data/validation_images/landmask',
                    process = True,
                    suppress_file_outputs = False,
                    fsd = False, 
                    algorithm_name='original_ift',
                    threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': 0,
                        'solidity_threshold': 0,
                        'tc_intensity_thresholds': (0, 0, 0),
                        'fc_intensity_thresholds': (0, 0, 0)})

    filter_pix, filter_floe, filter_fsd_plot = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                    validation_path= '../data/validation_images/labeled_floes_png', 
                    land_mask_path= '../data/validation_images/landmask',
                    process = True,
                    suppress_file_outputs = False,
                    fsd = False,
                    algorithm_name='filtered_ift',
                    threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': 0.55,
                        'solidity_threshold': 0.8,
                        'tc_intensity_thresholds': (0, 0, 0),
                        'fc_intensity_thresholds': (0, 0, 0)})
        
    param_bar_chart({"Original IFT": original_pix, 
                    "Filtered IFT": filter_pix}, 
                    out_dir='./algorithm_comparisons')
    

def prcurve():

    precisions_all = []
    recalls_all = []
    labels = []

    precisions = []
    recalls = []

    for i in range(100, 3000, 200):

        original_pix, original_floe, _ = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                        validation_path= '../data/validation_images/labeled_floes_png', 
                        land_mask_path= '../data/validation_images/landmask',
                        process = True,
                        suppress_file_outputs = True,
                        fsd = False, 
                        algorithm_name='original_ift',
                        threshold_params={'min_area': 100, 'max_area': i, 'circ_threshold': 0,
                            'solidity_threshold': 0,
                            'tc_intensity_thresholds': (0, 0, 0),
                            'fc_intensity_thresholds': (0, 0, 0)})

        precisions.append(original_floe['Floe PPV'])
        recalls.append(original_floe['Floe TPR'])
    labels.append('PR Curve, varying max area parameter')

    precisions_all.append(precisions)
    recalls_all.append(recalls)

    precisions = []
    recalls = []

    for i in range(0, 19, 1):

        original_pix, original_floe, _ = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                        validation_path= '../data/validation_images/labeled_floes_png', 
                        land_mask_path= '../data/validation_images/landmask',
                        process = True,
                        suppress_file_outputs = True,
                        fsd = False, 
                        algorithm_name='original_ift',
                        threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': (i/20),
                            'solidity_threshold': 0,
                            'tc_intensity_thresholds': (0, 0, 0),
                            'fc_intensity_thresholds': (0, 0, 0)})

        precisions.append(original_floe['Floe PPV'])
        recalls.append(original_floe['Floe TPR'])
    labels.append('PR Curve, varying circularity parameter')

    precisions_all.append(precisions)
    recalls_all.append(recalls)

    precisions = []
    recalls = []

    for i in range(0, 19, 1):

        original_pix, original_floe, _ = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                        validation_path= '../data/validation_images/labeled_floes_png', 
                        land_mask_path= '../data/validation_images/landmask',
                        process = True,
                        suppress_file_outputs = True,
                        fsd = False, 
                        algorithm_name='original_ift',
                        threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': 0,
                            'solidity_threshold': (i/20),
                            'tc_intensity_thresholds': (0, 0, 0),
                            'fc_intensity_thresholds': (0, 0, 0)})

        precisions.append(original_floe['Floe PPV'])
        recalls.append(original_floe['Floe TPR'])
    labels.append('PR Curve, varying solidity parameter')

    precisions_all.append(precisions)
    recalls_all.append(recalls)

    precisions = []
    recalls = []

    for i in range(0, 255, 10):

        original_pix, original_floe, _ = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                        validation_path= '../data/validation_images/labeled_floes_png', 
                        land_mask_path= '../data/validation_images/landmask',
                        process = True,
                        suppress_file_outputs = True,
                        fsd = False, 
                        algorithm_name='original_ift',
                        threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': 0,
                            'solidity_threshold': 0,
                            'tc_intensity_thresholds': (i, 0, 0),
                            'fc_intensity_thresholds': (0, 0, 0)})

        precisions.append(original_floe['Floe PPV'])
        recalls.append(original_floe['Floe TPR'])
    labels.append('PR Curve, varying TC channel 1 threshold')

    precisions_all.append(precisions)
    recalls_all.append(recalls)

    precisions = []
    recalls = []

    for i in range(0, 255, 10):

        original_pix, original_floe, _ = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                        validation_path= '../data/validation_images/labeled_floes_png', 
                        land_mask_path= '../data/validation_images/landmask',
                        process = True,
                        suppress_file_outputs = True,
                        fsd = False, 
                        algorithm_name='original_ift',
                        threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': 0,
                            'solidity_threshold': 0,
                            'tc_intensity_thresholds': (0, i, 0),
                            'fc_intensity_thresholds': (0, 0, 0)})

        precisions.append(original_floe['Floe PPV'])
        recalls.append(original_floe['Floe TPR'])
    labels.append('PR Curve, varying TC channel 2 threshold')

    precisions_all.append(precisions)
    recalls_all.append(recalls)

    precisions = []
    recalls = []

    for i in range(0, 255, 10):

        original_pix, original_floe, _ = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                        validation_path= '../data/validation_images/labeled_floes_png', 
                        land_mask_path= '../data/validation_images/landmask',
                        process = True,
                        suppress_file_outputs = True,
                        fsd = False, 
                        algorithm_name='original_ift',
                        threshold_params={'min_area': 100, 'max_area': 391**2, 'circ_threshold': 0,
                            'solidity_threshold': 0,
                            'tc_intensity_thresholds': (0, 0, i),
                            'fc_intensity_thresholds': (0, 0, 0)})

        precisions.append(original_floe['Floe PPV'])
        recalls.append(original_floe['Floe TPR'])
    labels.append('PR Curve, varying TC channel 3 threshold')

    precisions_all.append(precisions)
    recalls_all.append(recalls)

    generate_pr_curve(precisions_all, recalls_all, labels)



def param_bar_chart(outputs: list, out_dir: str):

    params = list(list(outputs.values())[0].keys())

    not_plotted = ['Pixel Pos. Likelihood Ratio', 'Pixel Neg. Likelihood Ratio', 'Pixel Diagnostic Odds Ratio', 'Pixel FB', 'Floe FB']
    for param in not_plotted:
        try:
           params.remove(param) 
        except ValueError:
            pass

    algos = outputs.keys()

    param_values = {}
    for algo in algos:
        param_values[algo] = [outputs[algo][param] for param in params]
        plt.plot(params, param_values[algo], marker='o', label=algo)
    

    plt.xlabel('Segmentation Performance Parameters')
    plt.ylabel('Values')
    plt.title('Figure X: Pixel-wise Image Analysis Parameters of Interest')
    plt.xticks(rotation=45, ha='right')

    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    plt.savefig(f"{out_dir}/param_comparison.png", dpi=300)

    plt.close()

    return

def generate_pr_curve(precisions_list, recalls_list, labels=None):

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    plt.suptitle('Figure X: PR curves varying threshold parameters')

    # Create a colormap
    cmap = plt.cm.get_cmap('viridis')

    for i, (precisions, recalls) in enumerate(zip(precisions_list, recalls_list)):
        row = i % 3
        col = i // 3
        colors = np.linspace(0, 1, len(precisions))  # Assigning different colors for each curve
        axs[row, col].scatter(recalls, precisions, c=colors, cmap=cmap)
        axs[row, col].set_xlabel('Recall')
        axs[row, col].set_ylabel('Precision')
        axs[row, col].set_title(labels[i] if labels else f'Curve {i+1}')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjusting the position of the title
    plt.savefig('prcurve_subplots.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
    #prcurve()