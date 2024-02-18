import h5py
import numpy as np
from pathlib import Path
from fastmri.data import transforms as T
from fastmri.data.subsample import create_mask_for_mask_type
from pygrappa import grappa
import torch
import time
import gc

folder_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/Preprocessed/multicoil_train/'
files = Path(folder_path).glob('**/*')
file_count = 1

mask_func = create_mask_for_mask_type('equispaced', [0.08], [4])

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()

def apply_grappa(masked_kspace, mask):
    """
    Applies GRAPPA algorithm
    References
    ----------
    [1] Griswold, Mark A., et al. "Generalized autocalibrating
       partially parallel acquisitions (GRAPPA)." Magnetic
       Resonance in Medicine: An Official Journal of the
       International Society for Magnetic Resonance in Medicine
       47.6 (2002): 1202-1210.
    Args:
        masked_kspace (torch.Tensor): Multi-coil masked input k-space of shape (num_coils, rows, cols, 2)
        mask (torch.Tensor): Applied mask of shape (1, 1, cols, 1)
    Returns:
        preprocessed_masked_kspace (torch.Tensor): Output of GRAPPA algorithm applied on masked_kspace
    """

    def get_low_frequency_lines(mask):
        l = r = mask.shape[-2] // 2
        while mask[..., r, :]:
            r += 1

        while mask[..., l, :]:
            l -= 1

        return l + 1, r

    l, r = get_low_frequency_lines(mask)
    num_low_freqs = r - l
    pad = (mask.shape[-2] - num_low_freqs + 1) // 2
    calib = masked_kspace[:, :, pad:pad + num_low_freqs].clone()
    preprocessed_masked_kspace = grappa(tensor_to_complex_np(masked_kspace), tensor_to_complex_np(calib), kernel_size=(5, 5), coil_axis=0)
    return to_tensor(preprocessed_masked_kspace)

for file in files:
    print(str(file_count)+". Starting to process file "+str(file)+'...')
    hf = h5py.File(file, 'a') # Open in append mode
    kspace = hf['kspace'][()]
    print("Shape of the raw kspace: ", str(np.shape(kspace)))
    kspace_torch = T.to_tensor(kspace)
    print("Shape of the torch tensor: ", str(kspace_torch.shape))
    masked_kspace, mask, _ = T.apply_mask(kspace_torch, mask_func)
    print("Shape of the generated mask: ", str(mask.shape))
    grappa_data = torch.zeros([masked_kspace.shape[0],masked_kspace.shape[1],masked_kspace.shape[2],masked_kspace.shape[3],masked_kspace.shape[4]])
    for slice in range(masked_kspace.shape[0]):
        grappa_data[slice,:,:,:,:] = apply_grappa(masked_kspace[slice,:,:,:,:], mask[0,:,:,:,:])
    print("Shape of the preprocessed grappa data: ", str(grappa_data.shape))
    grappa_data = tensor_to_complex_np(grappa_data)
    print("Shape of the numpy-converted grappa data: ", str(grappa_data.shape))
    # Add a key to the h5 file with grappa_data inside it
    hf.create_dataset('grappa_data', data=grappa_data)
    hf.close()
    time.sleep(1)
    del kspace, kspace_torch, masked_kspace, mask, grappa_data
    time.sleep(1)
    gc.collect()
    file_count += 1
    print('Done.')

