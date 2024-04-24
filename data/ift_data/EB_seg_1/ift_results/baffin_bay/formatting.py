import os

for file in [f for f in os.listdir('./') if not f.endswith('.py')]:
    
    for a in os.listdir(file + '/preprocess/hdf5-files'):
        if not a.endswith('.floes.tif'):
            os.remove(file + '/preprocess/hdf5-files/'+ a)