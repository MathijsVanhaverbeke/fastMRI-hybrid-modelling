import h5py
import numpy as np
from pathlib import Path
from fastmri.data import transforms as T
import torch
import time
import gc
import bart

folder_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/Preprocessed/multicoil_test/'
folder_path_full = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/Preprocessed/multicoil_test_full/'
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
    masked_kspace_T, mask = T.apply_mask(slice_kspace_T, mask_func)
    masked_kspace = T.tensor_to_complex_np(masked_kspace_T)
    return masked_kspace, mask

def generate_array(shape, R, tensor_out):
    length = shape[-1]

    # Initialize an array of zeros
    array = np.zeros(length)

    # Determine the central index
    array[length // 2] = 1

    # Set every R-1'th sample to 1, starting from the central index
    for i in range(length // 2, length, R):
        array[i] = 1

    # Mirror the behavior to the first half of the array
    if length % 2 == 0:
        array[1:length // 2] = np.flip(array[length // 2 + 1:])
    else:
        array[:length // 2] = np.flip(array[length // 2 + 1:])

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

def CG_SENSE(kspace, S, lamda=0.005, num_iter=50):
    ''' 
    Performs CG-SENSE reconstruction, i.e. CS reconstruction with a regular l2 norm for which the objective function then corresponds to a SENSE reconstruction.
    https://colab.research.google.com/github/mrirecon/bart-workshop/blob/master/mri_together_2023/bart_mritogether_2023.ipynb#scrollTo=kNWQGBaX9ISp

    Args:
        kspace (numpy.array): Slice kspace of shape (num_coils, rows, cols)
        S (numpy.array): Estimated sensitivity maps given by ESPIRiT of shape (num_coils, rows, cols)
        lamda: Value of the hyperparameter / regularizer of the l2 norm term
        num_iter: The amount of iterations the algorithm can run
    Returns:
        reconstruction (numpy.array): Estimated CG-SENSE reconstruction of shape (rows, cols))
    '''
    # Move coil axis to the back as expected by BART
    kspace_perm = np.moveaxis(kspace, 0, 2)
    S_perm = np.moveaxis(S, 0, 2)
    # Add extra dimension, because BART expects a 4D input array where the third dimension represents the batch size.
    kspace_perm = np.expand_dims(kspace_perm, axis=2)
    S_perm = np.expand_dims(S_perm, axis=2)
    # Perform CG-SENSE reconstruction
    reconstruction = bart.bart(1, 'pics -S -l2 -r {} -i {} -d 0'.format(lamda, num_iter), kspace_perm, S_perm)
    return reconstruction

def closer_to_4_or_8(float):
    diff_4 = np.abs(float - 4)
    diff_8 = np.abs(float - 8)

    if diff_4 < diff_8:
        return int(4)
    elif diff_8 < diff_4:
        return int(8)

for file in files:
    print(str(file_count)+". Starting to process file "+str(file)+'...')
    hf = h5py.File(file, 'r') # Open in read mode!
    nPE_mask = hf['mask'][()]
    sampled_columns = np.sum(nPE_mask)
    R = len(nPE_mask)/sampled_columns
    R = float(R)
    masked_kspace_ACS = hf['kspace'][()]
    print("Shape of the raw kspace: ", str(np.shape(masked_kspace_ACS)))
    hf = h5py.File(folder_path_full+file.name, 'r') # Open in read mode!
    kspace = hf['kspace'][()]
    mask = generate_array(kspace.shape, closer_to_4_or_8(R), tensor_out=False)
    masked_kspace = kspace * mask + 0.0
    sense_data = np.zeros((masked_kspace.shape[0], masked_kspace.shape[2], masked_kspace.shape[3]), dtype=np.complex64)
    for slice in range(masked_kspace.shape[0]):
        S = estimate_sensitivity_maps(masked_kspace_ACS[slice,:,:,:])
        sense_data[slice,:,:] = CG_SENSE(masked_kspace[slice,:,:,:], S)
    print("Shape of the numpy-converted sense data: ", str(sense_data.shape))
    hf = h5py.File(file, 'a') # Open in append mode!
    # Check if 'sense_data' key exists
    if 'sense_data' in hf:
        del hf['sense_data'] # Delete the existing dataset
    # Add a key to the h5 file with sense_data inside it
    hf.create_dataset('sense_data', data=sense_data)
    hf.close()
    time.sleep(1)
    del nPE_mask, masked_kspace_ACS, kspace, mask, masked_kspace, sense_data
    time.sleep(1)
    gc.collect()
    file_count += 1
    print('Done.')

