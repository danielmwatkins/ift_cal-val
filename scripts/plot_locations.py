import pandas as pd
import proplot as pplt
import cartopy.crs as ccrs
import numpy as np
import warnings
warnings.simplefilter('ignore')

locations = pd.read_csv('../data/site_locations.csv', index_col=0)

pplt.rc['reso'] = 'med'
pplt.rc['cartopy.circular'] = False

colors = {region: c['color'] for region, c in zip(
            locations.index,
            pplt.Cycle('dark2', len(locations)))}
locations['print_title'] = [c.replace('_', ' ').title().replace('Of', 'of') for c in locations.index]

fig, ax = pplt.subplots(width=4, proj='npstere', proj_kw={'lon_0': -45})
ax.format(land=True, color='k', boundinglat=50, landzorder=0, latmax=90)

linestyles = ['-', '-.', '--', '-', '-.', '--', '-.', '-', '--']

for site, ls, lat, lon in zip(locations.index, linestyles,
                                 locations.center_lat, locations.center_lon):

    xbox = np.array(locations.loc[site, ['left_x', 'left_x', 'right_x', 'right_x', 'left_x']].astype(float))
    ybox = np.array(locations.loc[site, ['lower_y', 'top_y', 'top_y', 'lower_y', 'lower_y']].astype(float))

    title = locations.loc[site, 'print_title']
    
    ax.plot(xbox, ybox, transform=ccrs.CRS('epsg:3413'), label=title, 
               color=colors[site], ls=ls, m='', zorder=5, lw=1.5)

        
ax.legend(loc='b', ncols=2,lw=1, order='F')
fig.save('../figures/fig01_region_map.png', dpi=300)