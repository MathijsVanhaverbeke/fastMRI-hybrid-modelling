"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import h5py
import numpy as np
from scipy import stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from fastmri.data import transforms


def evaluate(args, recons_key):

    metrics_1 = {}
    metrics_2 = {}
    MSE_1 = []
    MSE_2 = []
    NMSE_1 = []
    NMSE_2 = []
    PSNR_1 = []
    PSNR_2 = []
    SSIM_1 = []
    SSIM_2 = []

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, "r") as target, h5py.File(args.predictions_1_path / tgt_file.name, "r") as recons_1, \
        h5py.File(args.predictions_2_path / tgt_file.name, "r") as recons_2:
            if args.acquisition and args.acquisition != target.attrs["acquisition"]:
                continue

            if args.acceleration:
                filename = tgt_file.name
                mask_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/multicoil_test/'
                mask = h5py.File(os.path.join(mask_path,filename),'r')
                nPE_mask = mask['mask'][()]
                sampled_columns = np.sum(nPE_mask)
                R = len(nPE_mask)/sampled_columns
                R = float(R)
                if R > float(args.acceleration)+0.1 or R < float(args.acceleration)-0.1:
                    continue

            target = target[recons_key][()]
            recons_1 = recons_1["reconstruction"][()]
            recons_2 = recons_2["reconstruction"][()]
            target = transforms.center_crop(
                target, (target.shape[-1], target.shape[-1])
            )
            recons_1 = transforms.center_crop(
                recons_1, (target.shape[-1], target.shape[-1])
            )
            recons_2 = transforms.center_crop(
                recons_2, (target.shape[-1], target.shape[-1])
            )
            MSE_1.append(mse(target, recons_1).item())
            MSE_2.append(mse(target, recons_2).item())
            NMSE_1.append(nmse(target, recons_1).item())
            NMSE_2.append(nmse(target, recons_2).item())
            PSNR_1.append(psnr(target, recons_1).item())
            PSNR_2.append(psnr(target, recons_2).item())
            SSIM_1.append(ssim(target, recons_1).item())
            SSIM_2.append(ssim(target, recons_2).item())
    
    metrics_1['MSE'] = np.array(MSE_1)
    metrics_2['MSE'] = np.array(MSE_2)
    metrics_1['NMSE'] = np.array(NMSE_1)
    metrics_2['NMSE'] = np.array(NMSE_2)
    metrics_1['PSNR'] = np.array(PSNR_1)
    metrics_2['PSNR'] = np.array(PSNR_2)
    metrics_1['SSIM'] = np.array(SSIM_1)
    metrics_2['SSIM'] = np.array(SSIM_2)

    return metrics_1, metrics_2


def perform_pairwise_t_test(metrics_1, metrics_2):

    for (m_name_1, metric_1), (m_name_2, metric_2) in zip(metrics_1.items(), metrics_2.items()):
        t_statistic, p_value = stats.ttest_rel(metric_1, metric_2, nan_policy='raise', alternative='two-sided')
        print("For pairwise values of metric {}, the following results were obtained from a two-sided pairwise t-test: \
              T-statistic = {}, p-value = {}, degrees of freedom = {}.".format(m_name_1, str(t_statistic), str(p_value), str((len(metric_1)-1))))


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-1-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions of method 1",
    )
    parser.add_argument(
        "--predictions-2-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions of method 2",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        required=True,
        help="Which challenge",
    )
    parser.add_argument("--acceleration", type=int, default=None)
    parser.add_argument(
        "--acquisition",
        choices=[
            "CORPD_FBK",
            "CORPDFS_FBK",
            "AXT1",
            "AXT1PRE",
            "AXT1POST",
            "AXT2",
            "AXFLAIR",
        ],
        default=None,
        help="If set, only volumes of the specified acquisition type are used "
        "for evaluation. By default, all volumes are included.",
    )
    args = parser.parse_args()

    recons_key = (
        "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"
    )
    metrics_1, metrics_2 = evaluate(args, recons_key)
    perform_pairwise_t_test(metrics_1, metrics_2)
