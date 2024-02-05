import pandas as pd
import proplot as pplt
import pyproj
import numpy as np
import os
import xarray as xr
import warnings
import cartopy.crs as ccrs

warnings.simplefilter('ignore')
np.random.seed = 20230922

# Define the center coordinates and names for each region here
columns = ['location',
           'center_lat',
           'center_lon']

data = [['bering_strait', 65,  -170],
        ['beaufort_sea', 75,  -135],   
        ['hudson_bay', 60, -83],
        ['baffin_bay', 75, -65],        
        ['greenland_sea', 77,   -10],
        ['barents-kara_seas', 75,   54],        
        ['sea_of_okhostk', 58,   148],
        ['laptev_sea',  75, 125],
        ['chukchi-east_siberian_sea', 75,   166]]

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

locations = pd.DataFrame(data, columns=columns).set_index('location')
locations = locations.sort_index()

# Add colors using proplot cycle
colors = {region: c['color'] for region, c in zip(
            locations.index,
            pplt.Cycle('dark2', len(locations)))}
locations['color'] = [colors[r] for r in locations.index]

locations['center_x'] = np.nan
locations['center_y'] = np.nan

for region in locations.index:
    lon, lat = locations.loc[region, ['center_lon', 'center_lat']]
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('epsg:3413')
    transformer_xy = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
    x0, y0 = transformer_xy.transform(lon, lat)
    locations.loc[region, 'center_x'] = x0
    locations.loc[region, 'center_y'] = y0

locations['print_title'] = [c.replace('_', ' ').title().replace('Of', 'of') for c in locations.index]

locations_pretty = locations.copy()
locations_pretty.rename({'center_lat': 'Latitude',
                  'center_lon': 'Longitude'}, inplace=True, axis=1)
locations_pretty.set_index('print_title', drop=True, inplace=True)

print(locations_pretty.loc[:, ['Latitude', 'Longitude']].style.to_latex(hrules=True))

# Add columns for bounding box coordinates to the locations
# These will be used in defining the case boundaries.
new_columns = ['top_left_lat', 'top_left_lon', 'lower_left_lat', 'lower_left_lon',
               'top_right_lat', 'top_right_lon', 'lower_right_lat', 'lower_right_lon',
               'left_x', 'right_x', 'lower_y', 'top_y']
for c in new_columns:
    locations[c] = np.nan
locations['dx'] = np.nan
locations['dy'] = np.nan

corner_coords = find_box(center_lon=lon,
                         center_lat=lat,
                         length_x=xlength,
                         length_y=ylength)
for c in new_columns:
    locations.loc[site, c] = corner_coords[c]
    
# Save the names and mid points
locations.loc[:, ['center_lat', 'center_lon', 'center_x', 'center_y', 'left_x', 'right_x', 'lower_y', 'top_y']].to_csv('../data/site_locations.csv')