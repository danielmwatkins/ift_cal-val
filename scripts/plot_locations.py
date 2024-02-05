import pandas as pd
import proplot as pplt

locations = pd.read_csv('../data/site_locations.csv', index_col=0)

pplt.rc['reso'] = 'med'
fig, ax = pplt.subplots(width=4, proj='npstere', proj_kw={'lon_0': -45})
ax.format(land=True, color='k', boundinglat=50, landzorder=0)

linestyles = ['-', '-.', '--'] * 3
for site, color, ls, lat, lon in zip(locations.index, locations.color, linestyles,
                                 locations.center_lat, locations.center_lon):
    base_length=1500e3
    if site == 'baffin_bay':
        xlength = base_length * 0.75
        ylength = (base_length**2)/xlength
        
    elif site == 'greenland_sea':
        xlength = base_length * 0.9
        ylength = (base_length**2)/xlength

    else:
        xlength = base_length
        ylength = base_length
    xbox, ybox = draw_box(lon, lat, xlength, ylength)

    locations.loc[site, 'dx'] = xlength
    locations.loc[site, 'dy'] = ylength
            
    title = locations.loc[site, 'print_title']

    ax.plot(xbox, ybox, ls=ls, m='', color=color, zorder=5, lw=2,
                transform=ccrs.CRS('epsg:3413'), label=title)  
        
ax.legend(loc='b', ncols=2, title='Sites', lw=1, order='F')
fig.save('../figures/fig01_region_map.png', dpi=300)