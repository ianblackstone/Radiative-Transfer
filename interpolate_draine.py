import numpy as np
import os
from scipy import interpolate

a_min = 0.1
a_max = 0.3
num_a = 10


# Get a list of all files
Draine_list = os.listdir('Draine data/')

# get the number of files.
num_files = len(Draine_list)

# get the number of rows and columns in each file.
dummy_data = np.genfromtxt(os.path.join('Draine data/',Draine_list[0]), skip_header = 1)
num_rows = dummy_data.shape[0]
num_columns = dummy_data.shape[1]

# Don't need this in memory anymore.
dummy_data = []

initial_data = np.zeros((num_files,num_rows,num_columns))
a_Draine = np.zeros(num_files)

for i, file in enumerate(Draine_list):
    Draine_file = os.path.join('Draine data/',file)
    initial_data[i] = np.genfromtxt(Draine_file, skip_header=1)
    grain_size = file.split('_')[-2] + '.' +file.split('_')[-1]
    a_Draine[i] = float(grain_size)

a = np.linspace(a_min,a_max,num_a)

to_be_interpolated = np.where(np.logical_and(a_Draine >= a_min, a_Draine <= a_max))

to_be_interpolated = np.insert(to_be_interpolated, 0, to_be_interpolated[0]-1)
to_be_interpolated = np.append(to_be_interpolated, to_be_interpolated[-1]+1)

interpolated_data = np.empty(0)

interp_points = 5

