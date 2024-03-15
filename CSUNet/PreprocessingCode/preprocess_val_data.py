import h5py
import numpy as np
from pathlib import Path
from fastmri.data import transforms as T
from fastmri.data.subsample import EquispacedMaskFunc
import torch
import time
import gc
import bart
import scipy.io as sio

mat_file = sio.loadmat('/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/GitLab/master_thesis/fastMRI/sampling_profiles_CS.mat')
folder_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/Preprocessed/multicoil_val/'
files = Path(folder_path).glob('**/*')
file_count = 1

def apply_mask(slice_kspace, mask_func):
    ''' 
    Args:
        slice_kspace (numpy.array)
        mask_func (class)
    Returns:
        masked_kspace (numpy.array)
        mask (torch.tensor)
    '''
    slice_kspace_T = T.to_tensor(slice_kspace)
    masked_kspace_T, mask = T.apply_mask(slice_kspace_T, mask_func)   # Apply the mask to k-space
    masked_kspace = T.tensor_to_complex_np(masked_kspace_T)
    return masked_kspace, mask

def generate_array(shape, R, mat_file, tensor_out):
    if R == 4:
        array = mat_file['m320_CS4_mask'].squeeze()
    elif R == 8:
        array = mat_file['m320_CS8_mask'].squeeze()
    else:
        raise ValueError('Unrecognized acceleration factor specified. Must be 4 or 8.')
    # Make array compatible with fastmri mask function class
    for i in range(len(shape)-1):
        array = np.expand_dims(array, 0)
    if tensor_out:
        array = T.to_tensor(array)
    return array

def estimate_sensitivity_maps(kspace):
    ''' 
    Args:
        kspace (numpy.array): slice kspace of shape (num_coils, rows, cols)
    Returns:
        S (numpy.array): Estimated sensitivity maps given by ESPIRiT of shape (num_coils, rows, cols)
    '''
    # Move coil axis to the back as expected by BART
    kspace_perm = np.moveaxis(kspace, 0, 2)
    # Add extra dimension, because the ESPIRiT method expects a 4D input array where the third dimension represents the batch size.
    kspace_perm = np.expand_dims(kspace_perm, axis=2)
    # Estimate sensitivity maps with ESPIRiT method
    S = bart.bart(1, "ecalib -d0 -m1", kspace_perm)
    # Undo the previous operations to get the original data structure back
    S = np.moveaxis(S.squeeze(), 2, 0)
    return S

def CS(kspace, S, lamda=0.005, num_iter=50):
    ''' 
    Performs CS reconstruction
    https://mrirecon.github.io/bart/

    Args:
        kspace (numpy.array): Slice kspace of shape (num_coils, rows, cols)
        S (numpy.array): Estimated sensitivity maps given by ESPIRiT of shape (num_coils, rows, cols)
        lamda: Value of the hyperparameter / regularizer of the l1 norm term
        num_iter: The amount of iterations the algorithm can run
    Returns:
        reconstruction (numpy.array): Estimated CS reconstruction of shape (rows, cols))
    '''
    # Move coil axis to the back as expected by BART
    kspace_perm = np.moveaxis(kspace, 0, 2)
    S_perm = np.moveaxis(S, 0, 2)
    # Add extra dimension, because BART expects a 4D input array where the third dimension represents the batch size.
    kspace_perm = np.expand_dims(kspace_perm, axis=2)
    S_perm = np.expand_dims(S_perm, axis=2)
    # Perform CS
    reconstruction = bart.bart(1, 'pics -S -l1 -r {} -i {}'.format(lamda, num_iter), kspace_perm, S_perm)
    return reconstruction

for file in files:
    print(str(file_count)+". Starting to process file "+str(file)+'...')
    hf = h5py.File(file, 'a') # Open in append mode
    kspace = hf['kspace'][()]
    print("Shape of the raw kspace: ", str(np.shape(kspace)))
    mask_func = EquispacedMaskFunc(center_fractions=[0.08], accelerations=[4], seed=42) # Use seed for validation data
    masked_kspace_ACS, mask_ACS = apply_mask(kspace, mask_func)
    print("Shape of the generated ACS mask: ", str(mask_ACS.shape))
    mask = generate_array(kspace.shape, 4, mat_file, tensor_out=False)
    masked_kspace = kspace * mask + 0.0
    print("Shape of the generated CS mask: ", str(mask.shape))
    cs_data = np.zeros((kspace.shape[0], kspace.shape[2], kspace.shape[3]), dtype=np.complex64)
    for slice in range(kspace.shape[0]):
        S = estimate_sensitivity_maps(masked_kspace_ACS[slice,:,:,:])
        cs_data[slice,:,:] = CS(masked_kspace[slice,:,:,:], S)
    print("Shape of the numpy-converted CS data: ", str(cs_data.shape))
    # Add a key to the h5 file with cs_data inside it
    hf.create_dataset('cs_data', data=cs_data)
    hf.close()
    time.sleep(1)
    del kspace, masked_kspace, mask, cs_data
    time.sleep(1)
    gc.collect()
    file_count += 1
    print('Done.')

