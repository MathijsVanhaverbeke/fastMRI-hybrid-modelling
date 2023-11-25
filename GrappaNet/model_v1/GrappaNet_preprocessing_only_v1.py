## Set up a resource limiter, such that the script doesn't take up more than a certain amount of of RAM (normally 40GB is the limit). In that case, an error will be thrown

import resource

# Because micsd01 has very few jobs running currently, we can increase the RAM limit to a higher number than 40GB
resource.setrlimit(resource.RLIMIT_AS, (80_000_000_000, 80_000_000_000))


print('Resource limit set. Importing libraries...')


## Import libraries

import h5py, os
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft 
from utils import estimate_mdgrappa_kernel, calculate_mask, comp_sub_kspace, comp_img
import math
from pathlib import Path
from itertools import chain
import gc
import time


print('Libraries imported. Starting to prepare the dataset...')


## Prepare dataset

download_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI'
dicom_path = os.path.join(download_path,'fastMRI_brain_DICOM')
train_path = os.path.join(download_path,'multicoil_train')
validation_path = os.path.join(download_path,'multicoil_val')
test_path = os.path.join(download_path,'multicoil_test')
fully_sampled_test_path = os.path.join(download_path,'multicoil_test_full')

training_files = Path(train_path).glob('**/*')
validation_files = Path(validation_path).glob('**/*')
test_files = Path(test_path).glob('**/*')
fully_sampled_test_files = Path(fully_sampled_test_path).glob('**/*')
all_files = chain(training_files, validation_files, test_files, fully_sampled_test_files)

clustered_data_2 = np.load("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/exploration/16coil_slice_size_clustered_fastmri_data.npy", allow_pickle=True)
clustered_data_2 = clustered_data_2.item()

files_16_640_320 = clustered_data_2[(640,320)]
training_files = files_16_640_320

crop_size = (32,640,320)


print('All variables are loaded. Starting training dataset construction and GRAPPA kernel estimation...')


## Create training data pairs and estimate GRAPPA kernels

cnt = 1
last_mask = None
X_train = []
Y_train = []
grappa_wt = []
grappa_p = []

# Optional:
training_files = training_files[:25]   # Should result in a dataset comprised of around 400 multi-coil slices

for mri_f in sorted(training_files):
    filename = os.path.basename(mri_f)
    filename = filename.replace(".h5","")
    with h5py.File(mri_f,'r') as f:

        k = f['kspace'][()]
        sequence = f.attrs['acquisition']
        nSL, nCh, nFE, nPE = k.shape
        
        # Create the subsampled training data. In the GrappaNet paper, they say they performed experiments for R=4 and R=8, but they never mention the number or fraction of ACS lines used...
        mask = calculate_mask(nFE,nPE,0.08,4)
        last_mask = mask
        subsampled_k = k*mask
       
        for slice in range(nSL): 
            target_img = np.zeros((nCh,nFE,nPE),dtype=np.float)
            sub_kspace = np.zeros((nCh*2,nFE,nPE),dtype=np.float)
            wt,ps = estimate_mdgrappa_kernel(kspace=subsampled_k[slice,:,:,:],calib=None,kernel_size=(5,5),coil_axis=0) 
            for iCh in range(nCh):
                    target_img[iCh,:,:] = abs(fft.fftshift(fft.ifft2(k[slice,iCh,:,:])))
                    sub_kspace[iCh,:,:] = subsampled_k[slice,iCh,:,:].real
                    sub_kspace[iCh+nCh,:,:] = subsampled_k[slice,iCh,:,:].imag
            X_train.append(list(comp_sub_kspace(sub_kspace,crop_size)))
            Y_train.append(list(comp_img(target_img,(crop_size[0]//2,crop_size[1],crop_size[2]))))
            grappa_wt.append(wt)
            grappa_p.append(ps)
        
        print(cnt,filename,sequence,nSL,nCh,nFE,nPE,sub_kspace.shape)
        cnt += 1
        gc.collect()
        time.sleep(1)


print('Done. Saving results for other runs in the future...')


## Save the results as the previous code can run for a long time

import pickle

Y_train = np.array(Y_train).astype(np.float32)
X_train = np.array(X_train).astype(np.float32)

path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/mri/'
path_to_save_grappa_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/grappa/'

np.save(path_to_save_mri_data+"training_data_GrappaNet_16_coils.npy", X_train)
np.save(path_to_save_mri_data+"training_data_GT_GrappaNet_16_colis.npy", Y_train)

with open(path_to_save_grappa_data+'grappa_wt.pickle', 'wb') as handle:
    pickle.dump(grappa_wt, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_to_save_grappa_data+'grappa_p.pickle', 'wb') as handle:
    pickle.dump(grappa_p, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('Done.')

