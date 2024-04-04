"""Makes a map showing the 9 regions defined in the region_definitions table.
TBD: add the ice extent, if possible."""
import pandas as pd
import proplot as pplt
import cartopy.crs as ccrs
import numpy as np
import warnings
warnings.simplefilter('ignore')

regions = pd.read_csv('../data/metadata/region_definitions.csv', index_col=0)

pplt.rc['reso'] = 'med'
pplt.rc['cartopy.circular'] = False

colors = {region: c['color'] for region, c in zip(
            regions.index,
            pplt.Cycle('dark2', len(regions)))}
linestyles = {region: ls for region, ls in zip(regions.index,
                        ['-', '-.', '--', '-', '-.', '--', '-.', '-', '--'])}

regions['print_title'] = [c.replace('_', ' ').title().replace('Of', 'of') for c in regions.index]

fig, ax = pplt.subplots(width=4, proj='npstere', proj_kw={'lon_0': -45})
ax.format(land=True, color='k', boundinglat=50, landzorder=0, latmax=90)

for region, lat, lon in zip(regions.index, regions.center_lat, regions.center_lon):

    xbox = np.array(regions.loc[region, ['left_x', 'left_x', 'right_x', 'right_x', 'left_x']].astype(float))
    ybox = np.array(regions.loc[region, ['lower_y', 'top_y', 'top_y', 'lower_y', 'lower_y']].astype(float))

    title = regions.loc[region, 'print_title']
    
    ax.plot(xbox, ybox, transform=ccrs.CRS('epsg:3413'), label=title, 
               color=colors[region], ls=linestyles[region], m='', zorder=5, lw=1.5)

        
ax.legend(loc='b', ncols=2, lw=2, order='F')
fig.save('../figures/fig01_region_map.png', dpi=300)