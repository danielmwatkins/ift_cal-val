from floewise_img_process import floewise_img_process
from pixel_img_process import pixel_image_process
from tqdm import tqdm
from getimages import get_images
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

def cv_experiment():

    ift_path= '../data/ift_data/ift_pipeline_default/ift_results' 
    validation_path= '../data/validation_images/labeled_floes_png' 
    land_mask_path= '../data/validation_images/landmask'

    complete_cases = get_images(ift_path, validation_path, land_mask_path)

    print(complete_cases)

    complete_cases = complete_cases.to_dict().values()

    estimator = EstimatorExperiment()
    params = {'circ_threshold': [0.2, 0.3, 0.4, 0.5, 0.6]}
    clf = GridSearchCV(estimator, params)

    clf.fit(complete_cases)
    

    
class EstimatorExperiment(BaseEstimator, ClassifierMixin):

    def __init__(self, min_area = 100,
                        max_area = 90000,
                        circ_threshold = 0.6,
                        solidity_threshold = 0.8,
                        tc_intensity_thresholds = (0, 0, 0),
                        fc_intensity_thresholds = (0, 0, 0)):
        self.min_area = min_area
        self.max_area = max_area
        self.circ_threshold = circ_threshold
        self.solidity_threshold = solidity_threshold
        self.tc_intensity_thresholds = tc_intensity_thresholds
        self.fc_intensity_thresholds = fc_intensity_thresholds

    def fit(self, images):
        self.labels = []

        for row in images:

            case = {}

            floe_conf_mx, fps, fns, ift_to_man, intersections, labeled_image = floewise_img_process(row['manual_path'], row['ift_path'],  
                                                            row['land_mask_path'], row['tc_path'], row['fc_path'], 
                                                            threshold_params={'min_area': self.min_area, 'max_area': self.max_area, 'circ_threshold': self.circ_threshold, 
                                                            'solidity_threshold': self.solidity_threshold, 'tc_intensity_thresholds': self.tc_intensity_thresholds, 'fc_intensity_thresholds': self.fc_intensity_thresholds})

            pix_conf_mx = pixel_image_process(row['manual_path'], labeled_image, row['case_number'], 
                                        row['satellite'], str(row['land_mask_path']), 'ift', save_images=False)

            case.update(pix_conf_mx)
            case.update(floe_conf_mx)

            self.labels.append(case)

        return self


    def predict(self, images):
        self.prediction = []

        for row in images:

            case = {}

            floe_conf_mx, fps, fns, ift_to_man, intersections, labeled_image = floewise_img_process(row['manual_path'], row['ift_path'],  
                                                            row['land_mask_path'], row['tc_path'], row['fc_path'], 
                                                            threshold_params={'min_area': self.min_area, 'max_area': self.max_area, 'circ_threshold': self.circ_threshold, 
                                                            'solidity_threshold': self.solidity_threshold, 'tc_intensity_thresholds': self.tc_intensity_thresholds, 'fc_intensity_thresholds': self.fc_intensity_thresholds})

            pix_conf_mx = pixel_image_process(row['manual_path'], labeled_image, row['case_number'], 
                                        row['satellite'], str(row['land_mask_path']), 'ift', save_images=False)

            case.update(pix_conf_mx)
            case.update(floe_conf_mx)

            self.prediction.append(case)

        return self.prediction

    def score(self, images, y):

        prediction = []

        for row in images:

            case = {}

            floe_conf_mx, fps, fns, ift_to_man, intersections, labeled_image = floewise_img_process(row['manual_path'], row['ift_path'],  
                                                            row['land_mask_path'], row['tc_path'], row['fc_path'], 
                                                            threshold_params={'min_area': self.min_area, 'max_area': self.max_area, 'circ_threshold': self.circ_threshold, 
                                                            'solidity_threshold': self.solidity_threshold, 'tc_intensity_thresholds': self.tc_intensity_thresholds, 'fc_intensity_thresholds': self.fc_intensity_thresholds})

            pix_conf_mx = pixel_image_process(row['manual_path'], labeled_image, row['case_number'], 
                                        row['satellite'], str(row['land_mask_path']), 'ift', save_images=False)

            case.update(pix_conf_mx)
            case.update(floe_conf_mx)

            prediction.append(case)

        tp = 0
        fp = 0
        fn = 0

        for image in prediction:
            tp += image['t_pos_floes']
            fp += image['f_pos_floes']
            fn += image['f_neg_floes']

        tpr = (tp + fn) and tp / (tp + fn) or 0
        ppv = (tp + fp) and tp / (tp + fp) or 0

        return (ppv + tpr) and (2 * ppv * tpr) / (ppv + tpr) or 0
    

# Generates CSV with pixel confusion matrix for each image.
def process_floes(ift_path, validation_path, land_mask_path, algo_name, threshold_params: dict = None, suppress_file_outputs: bool = True):
    # Run image processor

    complete_cases = get_images(ift_path, validation_path, land_mask_path)

    results = {}


    for _, row in tqdm(complete_cases.iterrows(), total=len(complete_cases)):


        case = str(row['case_number']) + "_" + row['satellite']

        case_dict = row.to_dict()


        floe_conf_mx, fps, fns, ift_to_man, intersections, labeled_image = floewise_img_process(row['manual_path'], row['ift_path'],  
                                                            row['land_mask_path'], row['tc_path'], row['fc_path'], threshold_params=threshold_params)

        pix_conf_mx = pixel_image_process(row['manual_path'], labeled_image, row['case_number'], 
                                    row['satellite'], str(row['land_mask_path']), algo_name, save_images=not suppress_file_outputs)

        

        case_dict.update(pix_conf_mx)
        case_dict.update(floe_conf_mx)
        

        case_dict['fp_floes'] = fps
        case_dict['fn_floes'] = fns
        case_dict['ift_to_man'] = ift_to_man
        case_dict['intersections'] = intersections

        results[case] = case_dict

    with open(f'process_results/out_{algo_name}.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    cv_experiment()