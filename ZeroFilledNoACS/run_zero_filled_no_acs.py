"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import xml.etree.ElementTree as etree
from argparse import ArgumentParser
from pathlib import Path

import h5py
from tqdm import tqdm

import random
import numpy as np

import fastmri
from fastmri.data import transforms
from fastmri.data.mri_data import et_query


def fifty_fifty():
    return random.random() < .5


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
        array = transforms.to_tensor(array)

    return array


def save_zero_filled(data_dir, out_dir, which_challenge):
    reconstructions = {}

    for fname in tqdm(list(data_dir.glob("*.h5"))):
        undersampling_bool = fifty_fifty()
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            kspace = hf["kspace"][()]
            if undersampling_bool:
                mask = generate_array(kspace.shape, 4, tensor_out=False)
            else:
                mask = generate_array(kspace.shape, 8, tensor_out=False)
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
