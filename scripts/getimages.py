import os
import pandas as pd
import sys


# Returns dataframe with paths for cases where IFT and manual floes have both been generated
def get_images(ift_path, validation_path, land_path):

    # Get images where the whole IFT process worked to generate the hdf5 image.
    locations = [x.name for x in os.scandir(ift_path)]

    file_lists = {}
    
    for location in locations:

        csv_filename = ''

        for root, subdirs, files in os.walk(ift_path + "/" + location):
            for filename in files:
                if "evaluation_table.csv" in filename:
                    csv_filename = filename

        try: 
            ift_csv_df = pd.read_csv(ift_path + "/" + location + "/" + csv_filename)
        except Exception as e:
            print(e)
            sys.exit(1)

        passed_files = []

        for index, row in ift_csv_df.iterrows():
            if row['extractH5'] == 'pass':
                passed_files.append(row['location'])

        file_lists[location] = passed_files

    # Get all images from the cca case overview where floe masks have been generated successfully
    try: 
        cca_csv_df = pd.read_csv('../data/cca_cases_overview.csv')
    except Exception as e:
        print(e)
        sys.exit(1)
    cca_csv_df = cca_csv_df.drop(cca_csv_df[cca_csv_df.floe_mask != 'yes'].index)

    # Get all cases where IFT has run successfully for an image which ALSO has masked floes
    ift_positive_dates = []
    complete_cases = pd.DataFrame()
    for location, filelist in file_lists.items():

        for file in filelist:
            loc, dim, start_date, end_date = file.split('-')

            start_date = start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:]
            xdim = int(dim.split('km')[0])

            mask = (cca_csv_df['region'] == loc) & (cca_csv_df['start_date'] == start_date) & (cca_csv_df['dx_km'] == xdim)
            result = cca_csv_df[mask]

            for index, row in result.iterrows():
                if row['satellite'] == 'aqua':
                    ift_positive_dates.append(ift_path + '/' + location + '/' + file + '/preprocess/hdf5-files/' +
                        [x for x in os.listdir(ift_path + '/' + location + '/' + file + '/preprocess/hdf5-files') if 'aqua' in x][0])
                else:
                    ift_positive_dates.append(ift_path + '/' + location + '/' + file + '/preprocess/hdf5-files/' +
                        [x for x in os.listdir(ift_path + '/' + location + '/' + file + '/preprocess/hdf5-files') if 'terra' in x][0])


            complete_cases = pd.concat([complete_cases, result])

    # Add file paths to dataframe
    manual_paths = []
    land_masks = []
    
    # Add manual files to DF
    for index, row in complete_cases.iterrows():

        manual_filename = [x for x in os.listdir(validation_path) if (x.startswith("{:03d}".format(row['case_number']) + '_') and x.endswith(row['satellite'] + '.png'))][0]
        landmask_filename = [x for x in os.listdir(land_path) if (x.startswith("{:03d}".format(row['case_number']) + '_') and x.endswith('landmask.tiff'))][0]

        manual_paths.append(validation_path + '/' + manual_filename)
        land_masks.append(land_path + '/' + landmask_filename)

    complete_cases.insert(19, "manual_path", manual_paths)
    
    #Add IFT files to df
    complete_cases.insert(20, "ift_path", ift_positive_dates)

    complete_cases.insert(21, "land_mask_path", land_masks)

    return complete_cases