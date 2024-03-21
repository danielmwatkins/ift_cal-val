from img_analysis import analyze_algo

IFT_RESULTS_PATH = '../data/ift_results'
VALIDATION_IMG_PATH = '../data/validation_images/labeled_floes_png'

LAND_MASK_PATH = '../data/validation_images/landmask'

def main():

    analyze_algo(IFT_RESULTS_PATH, VALIDATION_IMG_PATH, LAND_MASK_PATH, process=True, algorithm_name='original_ift')


if __name__ == '__main__':
    main()