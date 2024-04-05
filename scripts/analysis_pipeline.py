import numpy as np
from img_analysis import analyze_algo
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

IFT_RESULTS_PATH = '../data/ift_results'
VALIDATION_IMG_PATH = '../data/validation_images/labeled_floes_png'

LAND_MASK_PATH = '../data/validation_images/landmask'

def main():

    # precisions = []
    # recalls = []

    # for i in range(100, 700, 50):

    #     original_pix, original_floe = analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
    #                 validation_path= '../data/validation_images/labeled_floes_png', 
    #                 land_mask_path= '../data/validation_images/landmask',
    #                 process = True,
    #                 suppress_file_outputs = True,
    #                 fsd = False, 
    #                 algorithm_name='original_ift',
    #                 threshold_params={'min_area': i})

    #     precisions.append(original_floe['ppv'])
    #     recalls.append(original_floe['tpr'])

    # generate_pr_curve(precisions, recalls)

        
    pass

def generate_pr_curve(precisions, recalls):

    # Plot the scatter plot and line of best fit
    plt.scatter(recalls, precisions, label='Data')

    # Add labels and legend
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Scatter Plot')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    main()