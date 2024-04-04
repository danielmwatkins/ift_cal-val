"""Generate figure 2, the sea ice fraction climatology for the study regions."""
import pandas as pd
import proplot as pplt
import numpy as np
import warnings
warnings.simplefilter('ignore')

regions = pd.read_csv('../data/metadata/region_definitions.csv', index_col='region')
regions['print_title'] = [c.replace('_', ' ').title().replace('Of', 'of') for c in regions.index]

sic_timeseries = pd.read_csv('../data/metadata/daily_sea_ice_fraction.csv', index_col=0, parse_dates=True)

colors = {region: c['color'] for region, c in zip(
            regions.index,
            pplt.Cycle('dark2', len(regions)))}
linestyles = {region: ls for region, ls in zip(regions.index,
                        ['-', '-.', '--', '-', '-.', '--', '-.', '-', '--'])}

fig, ax = pplt.subplots(width=6, height=4)
for region in colors:
    ax.plot(sic_timeseries[region].groupby(sic_timeseries[region].index.dayofyear).median(),
            shadedata=[sic_timeseries[region].groupby(sic_timeseries[region].index.dayofyear).quantile(0.25),
                       sic_timeseries[region].groupby(sic_timeseries[region].index.dayofyear).quantile(0.75)],
            fadedata=[sic_timeseries[region].groupby(sic_timeseries[region].index.dayofyear).quantile(0.1),
                       sic_timeseries[region].groupby(sic_timeseries[region].index.dayofyear).quantile(0.9)],
            c=colors[region], lw=2, ls=linestyles[region], label=regions.loc[region, 'print_title'])
ax.legend(loc='b', ncols=3,lw=2, order='F')
ax.format(ylabel='Sea Ice Fraction', xlabel='Day of Year', ylim=(0, 1),
         ylocator=np.arange(0.1, 0.91, 0.2))
fig.save('../figures/fig02_sea_ice_fraction.png', dpi=300)