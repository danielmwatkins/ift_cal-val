"""Performs a simple evaluation of whether each step of the algorithm ran without errors by checking whether
the expected number of output files were created. Only designed to work for scenarios using the IFT-pipeline."""

import numpy as np
import os
import pandas as pd

results_loc = '../data/ift_data/'
case_loc = '../data/ift_data/ift_case_definitions/'
scenarios = ['ift_pipeline_default', 'ift_pipeline_minarea_100px']
regions = ['baffin_bay', 'beaufort_sea', 'barents-kara_seas', 'chukchi-east_siberian_sea',
        'greenland_sea', 'hudson_bay', 'laptev_sea', 'sea_of_okhostk']

for scenario in scenarios:
    save_loc = results_loc + scenario + '/eval_tables/'
    try:
        os.mkdir(save_loc)
    except OSError as error:
        # if the folder exists already, keep going
        pass
    for region in regions:
        # get list of cases
        df_cases = pd.read_csv(case_loc + region + '_100km_cases.csv', index_col='location')

        # could add step here to add case number to the file if necessary

        steps = ['soit', 'landmasks', 'preprocess', 'extractH5', 'tracker']
        for step in steps:
            df_cases[step] = 'NA'
        
        for case in df_cases.index:
            if case in os.listdir(os.path.join(results_loc, scenario, 'ift_results', region)):
                case_path = os.path.join(results_loc, scenario, 'ift_results', region, case)
                
                # check soit successes    
                if 'soit' in os.listdir(case_path):
                    if len(os.listdir(case_path + '/soit' )) > 0:
                        df_cases.loc[case, 'soit'] = 'pass'
                    else:
                        df_cases.loc[case, 'soit'] = 'fail'
    
                # landmask successes
                if 'landmasks' in os.listdir(case_path):
                    if len(os.listdir(case_path + '/landmasks' )) > 0:
                        df_cases.loc[case, 'landmasks'] = 'pass'
                    elif df_cases.loc[case, 'soit'] == 'pass':
                        df_cases.loc[case, 'landmasks'] = 'fail'
    
    
                # for preprocess, need to look for files inside the hdf5 folder 
                files = [x for x in os.listdir(case_path + '/preprocess/') if x not in ['.DS_Store', 'hdf5-files']]
                if len(files) > 0:
                    df_cases.loc[case, 'preprocess'] = 'pass'
                    h5files = [x for x in os.listdir(case_path + '/preprocess/hdf5-files') if x != '.DS_Store']
            
                    # Check h5 and tracker if it passes the preprocess step
                    if len(h5files) > 0:
                        df_cases.loc[case, 'extractH5'] = 'pass'
                    else:
                        df_cases.loc[case, 'extractH5'] = 'fail'
                    trfiles = [x for x in os.listdir(case_path + '/tracker') if x != '.DS_Store']            
                    if len(trfiles) > 0:
                        df_cases.loc[case, 'tracker'] = 'pass'
                    else:
                        df_cases.loc[case, 'tracker'] = 'fail'            
            
                elif df_cases.loc[case, 'soit'] == 'pass':
                    if df_cases.loc[case, 'landmasks'] == 'pass': 
                        df_cases.loc[case, 'preprocess'] = 'fail'
        if np.any(df_cases['soit'] != 'NA'):
            # simple check of whether any data was present, since soit works 99% of the time
            df_cases.to_csv(save_loc + region + '_evaluation_table.csv')            