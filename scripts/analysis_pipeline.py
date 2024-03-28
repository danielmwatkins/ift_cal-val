import numpy as np
from img_analysis import analyze_algo
import matplotlib.pyplot as plt

IFT_RESULTS_PATH = '../data/ift_results'
VALIDATION_IMG_PATH = '../data/validation_images/labeled_floes_png'

LAND_MASK_PATH = '../data/validation_images/landmask'

def main():

    original_pix, original_floe = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                validation_path= '../data/validation_images/labeled_floes_png', 
                land_mask_path= '../data/validation_images/landmask',
                process=True, 
                algorithm_name='original_ift')

    eb_pix, eb_floe = analyze_algo(ift_path= '../data/ift_data/EB_seg/ift_results', 
                validation_path= '../data/validation_images/labeled_floes_png', 
                land_mask_path= '../data/validation_images/landmask',
                process=True, 
                algorithm_name='ellen_algo')

    original_p = {'name': 'original ift', 'ppv': original_floe['ppv'], 'tpr': original_floe['tpr']}
    eb_p = {'name': 'EB algo', 'ppv':eb_floe['ppv'], 'tpr': eb_floe['tpr']}

    # generate_pr_curve([original_p, min_100_p, eb_p])

    

def generate_pr_curve(values):

    name, x, y = [case['name'] for case in values], [case['ppv'] for case in values], [case['tpr'] for case in values]

    colors = np.random.rand(len(x))

    plt.figure(figsize=(5, 5))

    scatter = plt.scatter(x, y, c=colors)

    color_bar = plt.colorbar(scatter, ticks=np.linspace(0, 1, len(name)))
    color_bar.set_ticklabels(name)

    # Add labels and a title
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.title('Scatter Plot with Custom Colored Points')

    # Show the plot
    plt.savefig('./out.png')

    plt.close()


if __name__ == '__main__':
    main()