"""Sets up location specification tables for running the IFT-pipeline. This version is based on the table with
the 100 km cases. When the full validation table is updated, this code will need to be updated to use it.
"""

import pandas as pd
import pyproj
import numpy as np

overview = pd.read_csv('../data/cca_cases_overview.csv')
overview = overview.loc[overview.satellite == 'aqua'] # Only need one row per date, the pipeline will download both aqua and terra
overview['location'] = ''
overview.rename({'start_date': 'startdate'}, axis=1, inplace=True)
overview['dx'] = 100 # km
overview['dy'] = 100 # km

def draw_box(center_lon, center_lat, length_x, length_y, return_coords='stere'):
    """Return the corner coordinates for a box centered at center_lon, center_lat
    with side lengths given by length_x and length_y."""
    
    data = find_box(center_lon, center_lat, length_x, length_y, return_coords)

    if return_coords=='stere':
        return([data['left_x'], data['left_x'], data['right_x'], data['right_x'], data['left_x']],
               [data['lower_y'], data['top_y'], data['top_y'], data['lower_y'], data['lower_y']])
    else:
        return (data[['lower_left_lon', 'top_left_lon', 'top_right_lon', 'lower_right_lon', 'lower_left_lon']],
                data[['lower_left_lat', 'top_left_lat', 'top_right_lat', 'lower_right_lat', 'lower_left_lat']])

def find_box(center_lon, center_lat, length_x, length_y, return_coords='stere'):
    """Find the coordinates of each corner for a box centered at center_lon, center_lat with
    side lengths in meters specified by length_x and length_y. If return_coords is stere, then
    use the NSIDC polar stereographic, otherwise return latitude and longitude."""
    
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('epsg:3413')
    transformer_xy = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
    transformer_ll = pyproj.Transformer.from_crs(crs1, crs_to=crs0, always_xy=True)
    center_x, center_y = transformer_xy.transform(center_lon, center_lat)
    dx = length_x/2
    dy = length_y/2
    left = center_x - dx
    right = center_x + dx
    top = center_y + dy
    bottom = center_y - dy
    
    topleft_lon, topleft_lat = transformer_ll.transform(left, top)
    topright_lon, topright_lat = transformer_ll.transform(right, top)
    bottomleft_lon, bottomleft_lat = transformer_ll.transform(left, bottom)
    bottomright_lon, bottomright_lat = transformer_ll.transform(right, bottom)
    
    return pd.Series([topleft_lat, topright_lat, bottomleft_lat, bottomright_lat,
                      topleft_lon, topright_lon, bottomleft_lon, bottomright_lon,
                      left, right, bottom, top],
                      index=['top_left_lat', 'top_right_lat', 'lower_left_lat', 'lower_right_lat',
                             'top_left_lon', 'top_right_lon', 'lower_left_lon', 'lower_right_lon',
                             'left_x', 'right_x', 'lower_y', 'top_y']).round(5)

# Add columns for bounding box coordinates to the locations
# These will be used in defining the case boundaries.
new_columns = ['top_left_lat', 'top_left_lon', 'lower_left_lat', 'lower_left_lon',
               'top_right_lat', 'top_right_lon', 'lower_right_lat', 'lower_right_lon',
               'left_x', 'right_x', 'lower_y', 'top_y']

for c in new_columns:
    overview[c] = np.nan

for idx in overview.index:
    lon = overview.loc[idx, 'center_lon']
    lat = overview.loc[idx, 'center_lat']
    dx = overview.loc[idx, 'dx']
    dy = overview.loc[idx, 'dy']
    xlength = dx*1e3
    ylength = dy*1e3
    corner_coords = find_box(center_lon=lon,
                             center_lat=lat,
                             length_x=xlength,
                             length_y=ylength)
    for c in new_columns:
        overview.loc[idx, c] = corner_coords[c]

# Add end date column for 1 day later
end_dates = [(pd.to_datetime(x) + pd.to_timedelta('1D')).strftime('%Y-%m-%d') for x in overview.startdate]
overview['enddate'] = end_dates

# Generate a case_id that is unique to each sample
for idx in overview.index:
    dx = overview.loc[idx, 'dx']
    dy = overview.loc[idx, 'dy']
    imsize = str(dx) + 'km_by_' + str(dy) + 'km'
    region = overview.loc[idx, 'region']
    startdate = overview.loc[idx, 'startdate']
    enddate = overview.loc[idx, 'enddate']
    case_id = '-'.join([region, imsize, startdate.replace('-', ''), enddate.replace('-', '')])
    overview.loc[idx, 'location'] = case_id

# Finally, format to the specifications needed for IFT-pipeline
columns = ['location', 'center_lat', 'center_lon', 'top_left_lat', 'top_left_lon',
           'lower_right_lat', 'lower_right_lon', 'left_x', 'right_x', 'lower_y',
           'top_y', 'startdate', 'enddate']
for region, group in overview.groupby('region'):
    overview.loc[overview.region==region, columns].to_csv(
        '../data/location_specifications/' + region + '_100km_cases.csv', index=False)