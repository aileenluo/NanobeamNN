# Script to generate simulated diffraction library of a zone-plate focused beam diffracted by an epitaxial thin film
# Method developed by Tao Zhou (tzhou@anl.gov)
# Simulate over strain, tilt_lr, and tilt_ud, in that order, for 11.7 nm SrIrO3 on LSAT

from math import *
import numpy as np
import os
import time
from joblib import Parallel, delayed

upsampling1 = 1

energy = 11.3 # in keV
wavelength = 12.398/energy
K=2*pi/wavelength
c = 4.013 # Lattice constant in Angstroms
l = 2 # Diffraction order
alf0 = asin(wavelength*l/2/c)
alf = alf0 
twotheta = (2 * alf0) 
X0 =  256 # x pixel value for correct twotheta (requires detector calibration)
Xcen = 256 # x pixel value for center of ROI

distance = 0.85 # in meters
pixelsize = 55e-6/upsampling1*2 # The *2 is for downsampling the detector space 

gam0 = twotheta-alf

focal_length = 21.874e-3 # diameter * outermost zone width / wavelength
outer_angle = 149e-6/2/focal_length # diameter of FZP is 149 um
inner_angle = 77e-6/2/focal_length # diameter of CS is 77 um 

precision = 5e-4 # for fast numerical integration

det_x = np.arange(64*upsampling1).astype(np.float64)
det_y = np.arange(64*upsampling1).astype(np.float64)
det_x = det_x - det_x.mean() + X0 - (X0-Xcen)*upsampling1
det_y -= det_y.mean()

det_xx, det_yy = np.meshgrid(det_x,det_y)

gam = np.arcsin((det_xx-X0)*pixelsize/distance)+gam0

#detector
det_Qx = K*(np.cos(alf)-np.cos(gam))
det_Qz = K*(np.sin(gam)+np.sin(alf))
det_Qy = det_yy*pixelsize/distance*K

upsampling2 = 2

O_x = np.arange(64*upsampling2).astype(np.float64)
O_y = np.arange(64*upsampling2).astype(np.float64)
O_x -= O_x.mean()
O_y -= O_y.mean()

O_xx, O_yy = np.meshgrid(O_x,O_y)
O_xx = O_xx[:,:,np.newaxis,np.newaxis]
O_yy = O_yy[:,:,np.newaxis,np.newaxis]

# origin of the reciprocal space
O_Qx = -O_xx*pixelsize*upsampling1/upsampling2/distance*K*sin(alf)
O_Qz = O_xx*pixelsize*upsampling1/upsampling2/distance*K*cos(alf) # the sign of Qx and Qz are opposite in this convention
O_Qy = O_yy*pixelsize*upsampling1/upsampling2/distance*K
O_angle = np.sqrt(O_yy**2+O_xx**2)*pixelsize*upsampling1/upsampling2/distance
O_donut = (O_angle < outer_angle) * (O_angle > inner_angle)

folder = '/YOUR/DATA/SAVING/PATH'
strain_list = np.linspace(-0.005, 0.005, 41)
thickness = 117
print(strain_list)
t0 = time.time()

def sim_strain(strain):
    """Function to iterate over tilt_lr and tilt_ud for a specific strain quantity"""
    print('Time since start (hrs): ', (time.time()-t0)/3600)
    sim_mat = np.zeros((41, 41, 64, 64))
    for p0 in range(sim_mat.shape[0]):
        tilt_lr = (p0-20)*0.0025
        tilt_ud = np.linspace(-0.1, 0.1, 41)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        sim_mat[p0] = ((thickness*np.sinc(thickness*(det_Qz-2*pi/c*l/(1+strain)-O_Qz)/pi/2)**2 *\
                      (np.abs(det_Qx+2*pi/c*l*np.radians(tilt_lr)-O_Qx)<precision) *\
                      (np.abs(det_Qy+2*pi/c*l*np.radians(tilt_ud)-O_Qy)<precision))*O_donut)\
                      .sum(axis=(1,2)).reshape(-1, 64,upsampling1,64,upsampling1).mean(axis=(2,4))
    file_name = 'strain' + str(int(np.around(strain, 5)*1e5)) + '.npy'
    print(file_name)
    np.save(os.path.join(folder, file_name), sim_mat)

# Vectorization and parallelization must be adjusted according to the available RAM of your machine
Parallel(n_jobs=21, verbose=10)(delayed(sim_strain)(strain) for strain in strain_list)

print('Combining files')

sim_mat = np.zeros((41, 41, 41, 64, 64))
for i in range(strain_list.shape[0]):
    file_name = 'strain' + str(int(round(strain_list[i], 5)*1e5)) + '.npy'
    sim_mat[i] = np.load(os.path.join(folder, file_name))
    
np.save(os.path.join(folder, 'sim_mat.npy'), sim_mat)