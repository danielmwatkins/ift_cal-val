import os
import pandas as pd
import sys


# Returns dataframe with paths for cases where IFT and manual floes have both been generated
def get_images(ift_path, validation_path, land_path):

    # Get images where the whole IFT process worked to generate the hdf5 image.
    locations = [x.name for x in os.scandir(ift_path)]

    file_lists = {}
    
    for location in locations:

        eval_table_dir = ift_path + '/../eval_tables/'

        try:
            csv_filename = eval_table_dir + [x for x in os.listdir(eval_table_dir) if location in x][0]
            ift_csv_df = pd.read_csv(csv_filename)
        except IndexError as e:
            print(f"No evaluation table found for {location}")
            continue
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

        passed_files = []

        for _, row in ift_csv_df.iterrows():
            if row['extractH5'] == 'pass':
                passed_files.append(row['location'])

        file_lists[location] = passed_files

    # Get all images from the cca case overview where floe masks have been generated successfully
    try: 
        cca_csv_df = pd.read_csv(validation_path + '/../../validation_tables/qualitative_assessment_tables/all_100km_cases.csv')
    except Exception as e:
        print(e)
        sys.exit(1)

    # Get list of all validation masks
    manual_files = [file for file in os.listdir(validation_path) if file.endswith('.png')]

    complete_cases = pd.DataFrame()

    for location, filelist in file_lists.items():

        for file in filelist:

            # Deal with unpredictable use of hyphens in region name
            filename_parsed = file.split('-')
            start_date, xdim, loc = filename_parsed[-2], filename_parsed[-3], filename_parsed[:len(filename_parsed)-3]
            loc = '-'.join(loc)
            startdate = start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:]
            xdim = int(xdim.split('km')[0])

            mask = (cca_csv_df['region'] == loc) & (cca_csv_df['start_date'] == startdate) & (cca_csv_df['dx_km'] == xdim)
            result = cca_csv_df[mask]

            for _, row in result.iterrows():
                case = row['case_number']
                satellite = row['satellite']

                potential_man_file = "{:03d}".format(case) + "_" + loc + "_" + start_date + "_" + satellite + "_labeled_floes.png"

                if potential_man_file in manual_files:
                    to_append = result[result['satellite'] == satellite]
                    potential_man_file = validation_path + "/" + potential_man_file
                    file_path = ift_path + f"/{location}/" + file + "/preprocess/hdf5-files/"
                    file_path += [x for x in os.listdir(file_path) if satellite in x][0]
                    landmask_path = land_path + "/{:03d}_".format(case) + loc + '_landmask.tiff'
                    to_append = to_append.assign(manual_path=[potential_man_file], ift_path=[file_path], land_mask_path=[landmask_path])
                    complete_cases = pd.concat([complete_cases, to_append])

    return complete_cases