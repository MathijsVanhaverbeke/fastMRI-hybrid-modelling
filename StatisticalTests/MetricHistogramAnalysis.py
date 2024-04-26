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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from fastmri.data import transforms

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

import torch
from torchvision.models import vgg19
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Lambda



def plot_arrays(array1, array2, metric, model_name_1, model_name_2, save_path):
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Flatten the 2D array of axes to a 1D array for easier indexing
    axs = axs.flatten()

    # Plot the boxplot and histogram for the first array
    sns.boxplot(array1, ax=axs[0])
    sns.histplot(array1, ax=axs[1], bins='auto') # Automatically determine the number of bins

    # Plot the boxplot and histogram for the second array
    sns.boxplot(array2, ax=axs[2])
    sns.histplot(array2, ax=axs[3], bins='auto') # Automatically determine the number of bins

    # Customize the plot
    axs[0].set_title(f'Boxplot for the distribution of metric {metric} for model {model_name_1}')
    axs[1].set_title(f'Histogram for the distribution of metric {metric} for model {model_name_1}')
    axs[2].set_title(f'Boxplot for the distribution of metric {metric} for model {model_name_2}')
    axs[3].set_title(f'Histogram for the distribution of metric {metric} for model {model_name_2}')

    # Adjust the layout to prevent titles from overflowing
    plt.tight_layout()

    # Construct the full save path
    full_path = save_path + metric + '/histogram_and_boxplots.png'

    # Display the plot
    #plt.show()

    # Save the plot to a file instead of displaying it
    fig.savefig(full_path)

    # Close the figure to free up memory
    plt.close(fig)


def plot_array_differences(array1, array2, metric, model_name_1, model_name_2, save_path):
    # Array of differences
    diff_array = array1-array2

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))

    # Plot the boxplot and histogram
    sns.boxplot(diff_array, ax=axs[0])
    sns.histplot(diff_array, ax=axs[1], bins='auto') # Automatically determine the number of bins

    # Customize the plot
    axs[0].set_title(f'Boxplot for the distribution of differences for metric {metric} for model {model_name_1}-{model_name_2}')
    axs[1].set_title(f'Histogram for the distribution of differences for metric {metric} for model {model_name_1}-{model_name_2}')

    # Adjust the layout to prevent titles from overflowing
    plt.tight_layout()

    # Construct the full save path
    full_path = save_path + metric + '/histogram_and_boxplots_of_differences.png'

    # Display the plot
    #plt.show()

    # Save the plot to a file instead of displaying it
    fig.savefig(full_path)

    # Close the figure to free up memory
    plt.close(fig)


def detect_outliers_iqr(data):
    # Function to detect outliers using IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices_lower = np.where((data < lower_bound))
    outlier_indices_upper = np.where((data > upper_bound))
    return outlier_indices_lower, outlier_indices_upper


def find_out_outliers(array1, array2, dict1, dict2, metric, model_name_1, model_name_2, save_path):
    outliers_idx_lower_model_1, outliers_idx_upper_model_1 = detect_outliers_iqr(array1)
    outliers_idx_lower_model_2, outliers_idx_upper_model_2 = detect_outliers_iqr(array2)
    model_1_files = list(dict1.keys())
    model_2_files = list(dict2.keys())
    outliers_lower_model_1 = [model_1_files[i] for i in list(outliers_idx_lower_model_1[0])]
    outliers_upper_model_1 = [model_1_files[i] for i in list(outliers_idx_upper_model_1[0])]
    outliers_lower_model_2 = [model_2_files[i] for i in list(outliers_idx_lower_model_2[0])]
    outliers_upper_model_2 = [model_2_files[i] for i in list(outliers_idx_upper_model_2[0])]
    np.save(save_path+metric+f'/outliers_low_{model_name_1}.npy', outliers_lower_model_1)
    np.save(save_path+metric+f'/outliers_high_{model_name_1}.npy', outliers_upper_model_1)
    np.save(save_path+metric+f'/outliers_low_{model_name_2}.npy', outliers_lower_model_2)
    np.save(save_path+metric+f'/outliers_high_{model_name_2}.npy', outliers_upper_model_2)
    ordered_dict_model_1 = dict(sorted(dict1.items(), key=lambda item: item[1]))
    ordered_dict_model_2 = dict(sorted(dict2.items(), key=lambda item: item[1]))
    np.save(save_path+metric+f'/ordered_file_metric_dict_{model_name_1}.npy', ordered_dict_model_1, allow_pickle=True)
    np.save(save_path+metric+f'/ordered_file_metric_dict_{model_name_2}.npy', ordered_dict_model_2, allow_pickle=True)


def evaluate(args, recons_key):

    MSE_1 = {}
    MSE_2 = {}
    NMSE_1 = {}
    NMSE_2 = {}
    PSNR_1 = {}
    PSNR_2 = {}
    SSIM_1 = {}
    SSIM_2 = {}
    VGG_1 = {}
    VGG_2 = {}
    SVD_1 = {}
    SVD_2 = {}

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
            MSE_1[tgt_file.name] = mse(target, recons_1).item()
            MSE_2[tgt_file.name] = mse(target, recons_2).item()
            NMSE_1[tgt_file.name] = nmse(target, recons_1).item()
            NMSE_2[tgt_file.name] = nmse(target, recons_2).item()
            PSNR_1[tgt_file.name] = psnr(target, recons_1).item()
            PSNR_2[tgt_file.name] = psnr(target, recons_2).item()
            SSIM_1[tgt_file.name] = ssim(target, recons_1).item()
            SSIM_2[tgt_file.name] = ssim(target, recons_2).item()
            VGG_1[tgt_file.name] = vgg_loss(target, recons_1).item()
            VGG_2[tgt_file.name] = vgg_loss(target, recons_2).item()
            SVD_1[tgt_file.name] = stacked_svd(target, recons_1).item()
            SVD_2[tgt_file.name] = stacked_svd(target, recons_2).item()

    MSE_1_array = np.array(list(MSE_1.values()))
    MSE_2_array = np.array(list(MSE_2.values()))
    NMSE_1_array = np.array(list(NMSE_1.values()))
    NMSE_2_array = np.array(list(NMSE_2.values()))
    PSNR_1_array = np.array(list(PSNR_1.values()))
    PSNR_2_array = np.array(list(PSNR_2.values()))
    SSIM_1_array = np.array(list(SSIM_1.values()))
    SSIM_2_array = np.array(list(SSIM_2.values()))
    VGG_1_array = np.array(list(VGG_1.values()))
    VGG_2_array = np.array(list(VGG_2.values()))
    SVD_1_array = np.array(list(SVD_1.values()))
    SVD_2_array = np.array(list(SVD_2.values()))

    plot_arrays(MSE_1_array, MSE_2_array, 'MSE', args.model1, args.model2, args.save_results_path)
    plot_arrays(NMSE_1_array, NMSE_2_array, 'NMSE', args.model1, args.model2, args.save_results_path)
    plot_arrays(PSNR_1_array, PSNR_2_array, 'PSNR', args.model1, args.model2, args.save_results_path)
    plot_arrays(SSIM_1_array, SSIM_2_array, 'SSIM', args.model1, args.model2, args.save_results_path)
    plot_arrays(VGG_1_array, VGG_2_array, 'VGG', args.model1, args.model2, args.save_results_path)
    plot_arrays(SVD_1_array, SVD_2_array, 'SVD', args.model1, args.model2, args.save_results_path)

    plot_array_differences(MSE_1_array, MSE_2_array, 'MSE', args.model1, args.model2, args.save_results_path)
    plot_array_differences(NMSE_1_array, NMSE_2_array, 'NMSE', args.model1, args.model2, args.save_results_path)
    plot_array_differences(PSNR_1_array, PSNR_2_array, 'PSNR', args.model1, args.model2, args.save_results_path)
    plot_array_differences(SSIM_1_array, SSIM_2_array, 'SSIM', args.model1, args.model2, args.save_results_path)
    plot_array_differences(VGG_1_array, VGG_2_array, 'VGG', args.model1, args.model2, args.save_results_path)
    plot_array_differences(SVD_1_array, SVD_2_array, 'SVD', args.model1, args.model2, args.save_results_path)

    find_out_outliers(MSE_1_array, MSE_2_array, MSE_1, MSE_2, 'MSE', args.model1, args.model2, args.save_results_path)
    find_out_outliers(NMSE_1_array, NMSE_2_array, NMSE_1, NMSE_2, 'NMSE', args.model1, args.model2, args.save_results_path)
    find_out_outliers(PSNR_1_array, PSNR_2_array, PSNR_1, PSNR_2, 'PSNR', args.model1, args.model2, args.save_results_path)
    find_out_outliers(SSIM_1_array, SSIM_2_array, SSIM_1, SSIM_2, 'SSIM', args.model1, args.model2, args.save_results_path)
    find_out_outliers(VGG_1_array, VGG_2_array, VGG_1, VGG_2, 'VGG', args.model1, args.model2, args.save_results_path)
    find_out_outliers(SVD_1_array, SVD_2_array, SVD_1, SVD_2, 'SVD', args.model1, args.model2, args.save_results_path)


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
        "--save-results-path",
        type=str,
        required=True,
        help="Path to the main folder where we can save results",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        required=True,
        help="Which challenge",
    )
    parser.add_argument("--acceleration", type=int, default=None)
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
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
    evaluate(args, recons_key)
    print('Analysis done. Results saved.')
