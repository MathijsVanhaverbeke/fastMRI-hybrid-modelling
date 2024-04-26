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

import torch
from torchvision.models import vgg19
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Lambda


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
    VGG_1 = []
    VGG_2 = []
    SVD_1 = []
    SVD_2 = []

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, "r") as target, h5py.File(args.predictions_1_path / tgt_file.name, "r") as recons_1, \
        h5py.File(args.predictions_2_path / tgt_file.name, "r") as recons_2:
            if args.acquisition and args.acquisition != target.attrs["acquisition"]:
                continue

            if args.acceleration:
                filename = tgt_file.name
                mask_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/Preprocessed/multicoil_test/'
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
            VGG_1.append(vgg_loss(target, recons_1).item())
            VGG_2.append(vgg_loss(target, recons_2).item())
            SVD_1.append(stacked_svd(target, recons_1).item())
            SVD_2.append(stacked_svd(target, recons_2).item())
    
    metrics_1['MSE'] = np.array(MSE_1)
    metrics_2['MSE'] = np.array(MSE_2)
    metrics_1['NMSE'] = np.array(NMSE_1)
    metrics_2['NMSE'] = np.array(NMSE_2)
    metrics_1['PSNR'] = np.array(PSNR_1)
    metrics_2['PSNR'] = np.array(PSNR_2)
    metrics_1['SSIM'] = np.array(SSIM_1)
    metrics_2['SSIM'] = np.array(SSIM_2)
    metrics_1['VGG'] = np.array(VGG_1)
    metrics_2['VGG'] = np.array(VGG_2)
    metrics_1['SVD'] = np.array(SVD_1)
    metrics_2['SVD'] = np.array(SVD_2)

    return metrics_1, metrics_2


def perform_pairwise_t_test(metrics_1, metrics_2):

    for (m_name_1, metric_1), (m_name_2, metric_2) in zip(metrics_1.items(), metrics_2.items()):
        print("Performing statistical analysis for metric: " + m_name_1)
        t_statistic, p_value_t = stats.ttest_rel(metric_1, metric_2, nan_policy='raise', alternative='two-sided')
        print("For pairwise values of metric {}, the following results were obtained from a two-sided paired t-test: T-statistic = {}, p-value = {}, degrees of freedom = {}.".format(m_name_1, str(t_statistic), str(p_value_t), str((len(metric_1)-1))))
        _, p_value_shapiro = stats.shapiro(metric_1-metric_2)
        if p_value_shapiro < 0.05:
            print("WARNING: the paired differences are likely not normally distributed, as shapiro's p-value equals: ", str(p_value_shapiro))
            _, p_value_w = stats.wilcoxon(metric_1, metric_2, nan_policy='raise', alternative='two-sided')
            print("For pairwise values of metric {}, the following result was obtained from a two-sided Wilcoxon signed-rank test: p-value = {}.".format(m_name_1, str(p_value_w)))
            if p_value_w < 0.05:
                print("The median of the differences between the paired observations is significantly different from zero. Let's confirm the direction of the difference through one-sided Wilcoxon signed-rank tests now.")
                _, p_value_w_greater = stats.wilcoxon(metric_1, metric_2, nan_policy='raise', alternative='greater')
                print("For pairwise values of metric {}, the following result was obtained from a one-sided Wilcoxon signed-rank test in the positive direction: p-value = {}.".format(m_name_1, str(p_value_w_greater)))
                _, p_value_w_less = stats.wilcoxon(metric_1, metric_2, nan_policy='raise', alternative='less')
                print("For pairwise values of metric {}, the following result was obtained from a one-sided Wilcoxon signed-rank test in the negative direction: p-value = {}.".format(m_name_1, str(p_value_w_less)))
        print()


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


def stacked_svd(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute the average number of Singular Values required 
    to explain 90% of the variance in the residual error maps 
    of the reconstruction
    """
    residual_error_map = (gt-pred)**2
    U, S, Vh = np.linalg.svd(residual_error_map, full_matrices=True)
    num_slices = S.shape[0]
    im_size = S.shape[-1]
    singular_values_1d = S.flatten()
    abs_core = np.abs(singular_values_1d)
    sorted_indices = abs_core.argsort()[::-1]
    sorted_core = abs_core[sorted_indices]

    total_variance = np.sum(np.abs(sorted_core))

    # Calculate the cumulative sum of singular values
    cumulative_sum = np.cumsum(np.abs(sorted_core))

    num_svs = np.where(cumulative_sum >= 0.9*total_variance)[0][0] + 1

    num_svs_average = num_svs / num_slices

    return num_svs_average / im_size


# Define the preprocessing steps for the VGG loss
preprocess = Compose([
    ToTensor(),
    CenterCrop((224, 224)), # Ensure the center part of the image is used
    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def vgg_loss(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute VGG loss metric."""
    # Load the pre-trained VGG19 model
    vgg = vgg19(pretrained=True).features

    # Remove the last max pooling layer to get the feature maps
    vgg = torch.nn.Sequential(*list(vgg.children())[:-1])

    # Initialize a list to store the losses for each image in the batch
    losses = []

    # Convert inputs to the expected pixel range for RGB networks
    gt = gt*255
    pred = pred*255

    # Loop over each image in the batch
    for gt_image, pred_image in zip(gt, pred):
        # Preprocess the images
        gt_image = preprocess(gt_image)
        pred_image = preprocess(pred_image)

        # Ensure the images are batched
        gt_image = gt_image.unsqueeze(0)
        pred_image = pred_image.unsqueeze(0)

        # Extract features
        gt_features = vgg(gt_image)
        pred_features = vgg(pred_image)

        # Calculate VGG loss for the current pair of images
        loss = torch.nn.functional.mse_loss(gt_features, pred_features)
        losses.append(loss)

    # Average the losses across all images in the batch
    avg_loss = torch.mean(torch.stack(losses))

    return avg_loss.detach().cpu().numpy()


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
