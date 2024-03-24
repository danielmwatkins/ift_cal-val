from img_analysis import analyze_algo

IFT_RESULTS_PATH = '../data/ift_results'
VALIDATION_IMG_PATH = '../data/validation_images/labeled_floes_png'

LAND_MASK_PATH = '../data/validation_images/landmask'

def main():

    analyze_algo(ift_path= '../data/ift_data/ift_pipeline_default/ift_results', 
                validation_path= '../data/validation_images/labeled_floes_png', 
                land_mask_path= '../data/validation_images/landmask',
                process=True, 
                algorithm_name='original_ift')

    analyze_algo(ift_path= '../data/ift_data/ift_pipeline_minarea_100px/ift_results', 
                validation_path= '../data/validation_images/labeled_floes_png', 
                land_mask_path= '../data/validation_images/landmask',
                process=True, 
                algorithm_name='minarea100_ift')

    analyze_algo(ift_path= '../data/ift_data/EB_seg/ift_results', 
                validation_path= '../data/validation_images/labeled_floes_png', 
                land_mask_path= '../data/validation_images/landmask',
                process=True, 
                algorithm_name='ellen_algo')


if __name__ == '__main__':
    main()