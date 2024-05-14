import os
import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from fastmri.data import transforms

import torch
from torchvision.models import vgg19
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Lambda


def determine_and_apply_mask(target, recons, tgt_file):
    reconstruction_sense_path_string = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Reconstructions/Sense/reconstructions/'
    reconstruction_CS_path_string = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Reconstructions/CS/reconstructions/'
    reconstruction_sense_path = pathlib.Path(reconstruction_sense_path_string) / tgt_file.name
    reconstruction_CS_path = pathlib.Path(reconstruction_CS_path_string) / tgt_file.name
    reconstruction_sense = h5py.File(reconstruction_sense_path, 'r')
    reconstruction_CS = h5py.File(reconstruction_CS_path, 'r')
    reconstruction_sense = reconstruction_sense['reconstruction']
    reconstruction_CS = reconstruction_CS['reconstruction']
    reconstruction_sense = np.array(reconstruction_sense)
    reconstruction_CS = np.array(reconstruction_CS)
    reconstruction_sense = transforms.center_crop(reconstruction_sense, (target.shape[-1], target.shape[-1]))
    reconstruction_CS = transforms.center_crop(reconstruction_CS, (target.shape[-1], target.shape[-1]))
    sense_bitmask = np.ones_like(reconstruction_sense)
    sense_bitmask = np.where(reconstruction_sense != 0, sense_bitmask, 0).astype(int)
    CS_bitmask = np.ones_like(reconstruction_CS)
    CS_bitmask = np.where(reconstruction_CS != 0, CS_bitmask, 0).astype(int)
    intersection_mask = CS_bitmask & sense_bitmask
    gt = np.where(intersection_mask == 1, target, 0)
    pred = np.where(intersection_mask == 1, recons, 0)
    return gt, pred


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


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    VGG=vgg_loss,
    SVD=stacked_svd,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )


def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, "r") as target, h5py.File(
            args.predictions_path / tgt_file.name, "r"
        ) as recons:
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
            recons = recons["reconstruction"][()]
            target = transforms.center_crop(
                target, (target.shape[-1], target.shape[-1])
            )
            recons = transforms.center_crop(
                recons, (target.shape[-1], target.shape[-1])
            )
            target, recons = determine_and_apply_mask(target, recons, tgt_file)
            metrics.push(target, recons)

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
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
    metrics = evaluate(args, recons_key)
    print(metrics)
