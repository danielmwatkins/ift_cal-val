import proplot as pplt
import pandas as pd
import os
import numpy as np

data = []
for file in os.listdir('../data/validation_tables/'):
    if '.csv' in file:
        data.append(pd.read_csv('../data/validation_tables/' + file))
data = pd.concat(data)
data['case_number'] = [str(x).zfill(3) for x in data['case_number']]
data['start_date'] = pd.to_datetime(data['start_date'])


fig, ax = pplt.subplots()

all_files = data.copy()
all_files['month'] = data['start_date'].dt.month
all_files.loc[:, 'visible_sea_ice'] = all_files.loc[:, 'visible_sea_ice'].where(all_files.loc[:, 'visible_sea_ice']=='yes')
all_files.loc[:, 'visible_landfast_ice'] = all_files.loc[:, 'visible_landfast_ice'].where(all_files.loc[:, 'visible_landfast_ice']=='yes')
all_files.loc[:, 'visible_floes'] = all_files.loc[:, 'visible_floes'].where(all_files.loc[:, 'visible_floes']=='yes')
all_files.rename({'cloud_fraction_manual': 'Number of images',
                  'visible_sea_ice': 'Sea ice',
                  'visible_floes': 'Sea ice floes',
                  'visible_landfast_ice': 'Landfast ice'}, axis=1, inplace=True)

ax.bar(all_files.loc[:, ['month', 'Number of images', 'Sea ice', 'Sea ice floes', 'Landfast ice']].groupby('month').count())
ax.legend(loc='b', ncols=2)
ax.format(ylabel='Count', xlabel='', xlocator=np.arange(3, 10),
          xformatter=['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
          xtickminor=False)
fig.save('../figures/data_availability.png', dpi=300)