import os
import numpy as np

data_source = 'Draine data SiC/'

Draine_list = os.listdir(data_source)

a = np.zeros(len(Draine_list))

for i, file in enumerate(Draine_list):
    
    Draine_file = os.path.join(data_source,file)
    
    # print(Draine_file)
    # Import the data from a file for astronomical silicate.
    # [ wavelength (microns) , Q_abs , Q_scatt , g = <cos(theta)> ]
   
    grain_size = file.split('_')[-2] + '.' +file.split('_')[-1]
    # print(grain_size)
    a[i] = float(grain_size)