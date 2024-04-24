"""MASIE is an NSIDC data product that uses multiple sensors to estimate the location of the sea ice edge. It is available at 4 km resolution from 2006-present, and 1-km resolution since 2014. We can use this data for a more acurate measure of the sea ice fraction within images.

Once the file is made, it can be run in the terminal using

wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off -i masie_urls.txt
"""

import pandas as pd
import numpy as np
all_cases = pd.read_csv('../data/validation_tables/qualitative_assessment_tables/all_100km_cases.csv')
dates = pd.to_datetime([d for d in np.unique(all_cases['start_date'])])

filepaths = []
for d in dates:
    if d.year >= 2006:
        prefix = 'https://noaadata.apps.nsidc.org/NOAA/G02186/geotiff/4km/ice_only'
        year = str(d.year)
        fname = 'masie_ice_r00_v01_{d}_4km.tif'.format(d=year + str(d.dayofyear).zfill(3))
        filepaths.append('/'.join([prefix, year, fname]))

with open("../../data/MASIE_images/masie_urls.txt", 'w') as file:
    for url in filepaths:
        file.write(url+'\n')