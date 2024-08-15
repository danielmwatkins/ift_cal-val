# The labeled floe images are produced in Photoshop and
# are designed to have the exact dimentions
# Naming issue: would be easier if the region names didn't have underscores

import rasterio
import os
import pandas as pd

# Example:
png_loc = '../data/validation_images/labeled_floes_png/'
save_loc = '../data/validation_images/labeled_floes_geotiff/'
ref_loc = '../data/modis/truecolor/'

labeled_images = [f for f in os.listdir(png_loc) if 'png' in f]

site_defs = pd.read_csv('../data/metadata/region_definitions.csv')
regions = list(site_defs['region'].values)

for lb_image in labeled_images:
    for region in regions:
        if region in lb_image:
            cn, reg, date, sat, _, _ = lb_image.replace(region, region.replace('_', '')).split('_')
            sat = sat.replace('.png', '')
            break
    ref_image = '_'.join([cn, region, '100km', date]) + '.' + sat + '.truecolor.250m.tiff'

    with rasterio.open(ref_loc + ref_image) as src_dataset:
        
        # Get a copy of the source dataset's profile. Thus our
        # destination dataset will have the same dimensions,
        # number of bands, data type, and georeferencing as the
        # source dataset.
        kwds = src_dataset.profile
    
        # Change the format driver for the destination dataset to
        # 'GTiff', short for GeoTIFF.
        kwds['driver'] = 'GTiff'
    
        # Add GeoTIFF-specific keyword arguments.
        kwds['tiled'] = True
        kwds['blockxsize'] = 256
        kwds['blockysize'] = 256
        kwds['photometric'] = 'MINISBLACK'
        kwds['compress'] = 'PNG'
    
        # Floe data only has 1 layer (binary data)
        kwds['count'] = 1

    labeled = rasterio.open(png_loc + lb_image)
    data = labeled.read(1)
    with rasterio.open(save_loc + lb_image.replace('png', 'tiff'), mode='w', **kwds) as new_file:
        new_file.write(data, 1)