import pandas as pd
import numpy as np
from skimage.measure import regionprops_table

# Advantage of doing this with regionprops_table is that we already use regionprops_table in IFT,
# so there are minimal changes needed to put a filter directly into the code
def ift_filter(labeled_image,
               truecolor_image,
               falsecolor_image,
               min_area = 100,
               max_area = (391**2),
               circ_threshold = 0,
               solidity_threshold = 1, # Ratio of area to convex area - measure of gaps. 
               tc_intensity_thresholds = (0, 0, 0), # No idea what the best numbers for these are! Also not sure if we want max or min. 0 means the threshold will always be exceeded.
               fc_intensity_thresholds = (0, 0, 0)): 
    """Filter function for use with the labeled image output from Ice Floe Tracker. The function uses the regionprops_table function from 
    scikit-image to compute metrics for labeled objects. Returns a filtered labeled image and a properties table with a columns
    'label', 'area', 'perimeter', 'solidity', 'tc_channel'[0-2], 'fc_channel'[0-2], and flag columns
    'area_flag', 'circularity_flag', 'solidity_flag', 'tc0_flag', 'tc1_flag', 'tc2_flag', 'fc0_flag', 'fc1_flag', 'fc2_flag', 'flagged'.
    The 'flagged' column is set to True if at least one of the flag columns is true. Flagged values failed at least one of the tests.
    """

    # other options: can look at intensity range, potentially also intensity standard deviation
    props = pd.DataFrame(regionprops_table(labeled_image, truecolor_image, properties=['label', 'area', 'perimeter', 'solidity', 'intensity_mean']))
    props_fc = pd.DataFrame(regionprops_table(labeled_image, falsecolor_image, properties=['label', 'intensity_mean']))
    
    props.rename({'intensity_mean-0': 'tc_channel0',
                  'intensity_mean-1': 'tc_channel1',
                  'intensity_mean-2': 'tc_channel2'}, axis=1, inplace=True)
    props_fc.rename({'intensity_mean-0': 'fc_channel0',
                     'intensity_mean-1': 'fc_channel1',
                     'intensity_mean-2': 'fc_channel2'}, axis=1, inplace=True)
    
    props = props.merge(props_fc, left_on='label', right_on='label')
    
    props['circularity'] = 4*np.pi*props['area']/props['perimeter']**2
    
    props['area_flag'] = (props['area'] <= min_area) | (props['area'] >= max_area)
    props['circularity_flag'] = props['circularity'] < circ_threshold
    props['solidity_flag'] = props['solidity'] < solidity_threshold
    for label, thresholds in zip(['tc', 'fc'], [tc_intensity_thresholds, fc_intensity_thresholds]):
        for channel in range(3):
            props[label + str(channel) + '_flag'] = props[label + '_channel' + str(channel)] < thresholds[channel]
    
    all_flags = ['area_flag', 'circularity_flag', 'solidity_flag', 'tc0_flag', 'tc1_flag', 'tc2_flag', 'fc0_flag', 'fc1_flag', 'fc2_flag']
    props['flagged'] = props.loc[:, all_flags].any(axis=1)

    result_image = np.copy(labeled_image)
    for label in props.loc[props['flagged'], 'label']:
        result_image[result_image == label] = 0

    return result_image, props