# Script to fit the maximum correlation between diffraction patterns to solve for strain, tilt_lr, and tilt_ud
# This method was developed by Tao Zhou (tzhou@anl.gov)

import os
import numpy as np
from math import *
import time
from skimage import io

# Data folder
folder = '/SOME_PATH/NanobeamNN/data'

# Load the experimental data
data = np.load(os.path.join(folder, 'exp_data.npy')

# Solve for parameters directly using the simulated diffraction library
sim_mat = np.load(os.path.join(folder, 'sim_mat.npy'))

sim_mat /= sim_mat.sum(axis=(3, 4), keepdims=True) # Normalize

strain = np.zeros((data.shape[0], ))
tilt_lr = np.zeros((data.shape[0], ))
tilt_ud = np.zeros((data.shape[0], ))

i = 0
t0 = time.time()

while i + 400 < data.shape[0]: # 400 is the number of data points to fit at once (vectorized operations for efficiency)
    
    # Fitting 400 points at once using the simulation library provided in this package requires ~1 TB RAM
    # Adjust the number of points according to your machine's available memory
    print('Current index: ', i)

    dd = np.copy(data)
    dd_sum = (dd[i:i+400, np.newaxis, np.newaxis, np.newaxis]*sim_mat).sum(axis=(4, 5))

    # Weighted sum and interpolation of center of mass for highest correlation
    dd_sum_s = dd_sum.sum(axis=(2, 3))
    dd_sum_s -= dd_sum_s.min(axis=1, keepdims=True)

    dd_sum_lr = dd_sum.sum(axis=(1, 3))
    dd_sum_lr -= dd_sum_lr.min(axis=1, keepdims=True)
               
    dd_sum_ud = dd_sum.sum(axis=(1, 2))
    dd_sum_ud -= dd_sum_ud.min(axis=1, keepdims=True)

    i_s = (dd_sum_s*np.arange(41)).sum(axis=(1))/dd_sum_s.sum(axis=(1))
    s = (i_s-20)*0.00025

    i_lr = (dd_sum_lr*np.arange(41)).sum(axis=(1))/dd_sum_lr.sum(axis=(1))
    tilt_lr = (i_lr-20)*0.0025
    
    i_ud = (dd_sum_ud*np.arange(41)).sum(axis=(1))/dd_sum_ud.sum(axis=(1))
    tilt_ud = (i_ud-20)*0.005

    print('Fit time (min.): ', (time.time()-t0)/60)
    strain[i:i+400] = s
    tilt_lr[i:i+400] = tilt_lr
    tilt_ud[i:i+400] = tilt_ud
    i += 400
    
print('Last batch starting index: ', i)

dd = np.copy(data)
dd_sum = (dd[i:, np.newaxis, np.newaxis, np.newaxis]*sim_mat).sum(axis=(4, 5))

dd_sum_s = dd_sum.sum(axis=(2, 3))
dd_sum_s -= dd_sum_s.min(axis=1, keepdims=True)

dd_sum_lr = dd_sum.sum(axis=(1, 3))
dd_sum_lr -= dd_sum_lr.min(axis=1, keepdims=True)

dd_sum_ud = dd_sum.sum(axis=(1, 2))
dd_sum_ud -= dd_sum_ud.min(axis=1, keepdims=True)

i_s = (dd_sum_s*np.arange(41)).sum(axis=(1))/dd_sum_s.sum(axis=(1))
s = (i_s-20)*0.00025

i_lr = (dd_sum_lr*np.arange(41)).sum(axis=(1))/dd_sum_lr.sum(axis=(1))
tilt_lr = (i_lr - 20)*0.0025

i_ud = (dd_sum_ud*np.arange(41)).sum(axis=(1))/dd_sum_ud.sum(axis=(1))
tilt_ud = (i_ud-20)*0.005

print('Fit time (min.): ', (time.time()-t0)/60)
strain[i:] = s
tilt_lr[i:] = tilt_lr
tilt_ud[i:] = tilt_ud

# Note that these files are included in the data made available with the paper if you want to avoid overriding them
np.save(os.path.join(folder, 'strain_fit.npy'), strain)
np.save(os.path.join(folder, 'tilt_lr_fit.npy'), tilt_lr)
np.save(os.path.join(folder, 'tilt_ud_fit.npy'), tilt_ud)