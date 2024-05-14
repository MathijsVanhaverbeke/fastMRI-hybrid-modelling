import xml.etree.ElementTree as etree
from argparse import ArgumentParser
from pathlib import Path

import h5py
from tqdm import tqdm

import numpy as np

import fastmri
from fastmri.data import transforms
from fastmri.data.mri_data import et_query

import scipy.io as sio

mat_file = sio.loadmat('/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/GitLab/master_thesis/fastMRI/sampling_profiles_CS.mat')


def closer_to_4_or_8(float):
    diff_4 = np.abs(float - 4)
    diff_8 = np.abs(float - 8)

    if diff_4 < diff_8:
        return int(4)
    elif diff_8 < diff_4:
        return int(8)


def find_original_R(fname):
    file_name = fname.name
    original_test_file = Path("/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/Preprocessed/multicoil_test/") / file_name
    hf = h5py.File(original_test_file, 'r') # Open in read mode!
    nPE_mask = hf['mask'][()]
    sampled_columns = np.sum(nPE_mask)
    R = len(nPE_mask)/sampled_columns
    R = float(R)
    return closer_to_4_or_8(R)


def generate_array(shape, R, mat_file, tensor_out):
    if R == 4:
        array = mat_file['m320_CS4_mask'].squeeze()
    elif R == 8:
        array = mat_file['m320_CS8_mask'].squeeze()
    else:
        raise ValueError('Unrecognized acceleration factor specified. Must be 4 or 8.')
    # Calculate padding needed to reach the desired length
    desired_length = shape[-1]
    padding_needed = desired_length - len(array)
    if padding_needed > 0:
        # Calculate padding width for symmetric padding
        padding_width = (padding_needed // 2, padding_needed - padding_needed // 2)
        # Pad the array symmetrically
        array = np.pad(array, padding_width, mode='symmetric')
    elif padding_needed < 0:
        # Calculate trimming indices for symmetric trimming
        trim_start = -padding_needed // 2
        trim_end = len(array) + padding_needed // 2
        # Trim the array symmetrically
        array = array[trim_start:trim_end]
    # Make array compatible with fastmri mask function class
    for i in range(len(shape)-1):
        array = np.expand_dims(array, 0)
    if tensor_out:
        array = transforms.to_tensor(array)
    return array


def save_zero_filled(data_dir, out_dir, which_challenge):
    reconstructions = {}

    for fname in tqdm(list(data_dir.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            kspace = hf["kspace"][()]
            mask = generate_array(kspace.shape, find_original_R(fname), mat_file, tensor_out=False)
            masked_kspace_np = kspace * mask + 0.0
            masked_kspace_np = masked_kspace_np.astype(np.complex64)
            masked_kspace = transforms.to_tensor(masked_kspace_np)

            # extract target image width, height from ismrmrd header
            enc = ["encoding", "encodedSpace", "matrixSize"]
            crop_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
            )

            # inverse Fourier Transform to get zero filled solution
            image = fastmri.ifft2c(masked_kspace)

            # check for FLAIR 203
            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            # crop input image
            image = transforms.complex_center_crop(image, crop_size)

            # absolute value
            image = fastmri.complex_abs(image)

            # apply Root-Sum-of-Squares if multicoil data
            if which_challenge == "multicoil":
                image = fastmri.rss(image, dim=1)

            reconstructions[fname.name] = image

    fastmri.save_reconstructions(reconstructions, out_dir)


def create_arg_parser():
    parser = ArgumentParser()

    # Here, data_path is the path to the fully sampled test kspace data
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the reconstructions to",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        required=True,
        help="Which challenge",
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    save_zero_filled(args.data_path, args.output_path, args.challenge)
