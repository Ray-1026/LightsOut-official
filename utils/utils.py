import os
import cv2
import numpy as np
import skimage
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.draw import disk


def IoU(pred, target):
    assert pred.shape == target.shape, "Prediction and target must have the same shape."

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def mean_IoU(y_true, y_pred, num_classes):
    """
    Calculate the mean Intersection over Union (mIoU) score.

    Args:
        y_true (np.ndarray): Ground truth labels (integer class values).
        y_pred (np.ndarray): Predicted labels (integer class values).
        num_classes (int): Number of classes.

    Returns:
        float: The mean IoU score across all classes.
    """
    iou_scores = []

    for cls in range(num_classes):
        # Create binary masks for the current class
        true_mask = y_true == cls
        pred_mask = y_pred == cls

        # Calculate intersection and union
        intersection = np.logical_and(true_mask, pred_mask)
        union = np.logical_or(true_mask, pred_mask)

        # Compute IoU for the current class
        if np.sum(union) == 0:
            # Handle edge case: no samples for this class
            iou_scores.append(np.nan)
        else:
            iou_scores.append(np.sum(intersection) / np.sum(union))

    # Calculate mean IoU, ignoring NaN values (classes without samples)
    mean_iou = np.nanmean(iou_scores)
    return mean_iou


def RGB2YCbCr(img):
    img = img * 255.0
    r, g, b = torch.split(img, 1, dim=0)
    y = torch.zeros_like(r)
    cb = torch.zeros_like(r)
    cr = torch.zeros_like(r)

    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    y = y / 255.0

    cb = -0.148 * r - 0.291 * g + 0.439 * b + 128
    cb = cb / 255.0

    cr = 0.439 * r - 0.368 * g - 0.071 * b + 128
    cr = cr / 255.0

    img = torch.cat([y, y, y], dim=0)
    return img


def extract_peaks(prob_map, thr=0.5, pool=7):
    """
    prob_map: (H, W) after sigmoid
    return: tensor of peak coordinates  [K, 2]  (x, y)
    """
    # binary mask
    pos = prob_map > thr

    # non‑maximum suppression
    nms = F.max_pool2d(
        prob_map.unsqueeze(0).unsqueeze(0),
        kernel_size=pool,
        stride=1,
        padding=pool // 2,
    )
    peaks = (prob_map == nms.squeeze()) & pos
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    return torch.stack([xs, ys], dim=1)  # (K, 2)


def pick_radius(radius_map, centers, ksize=3):
    """
    radius_map: (H, W) ∈ [0, 1]
    centers: (K, 2)  x,y
    return: (K,) radii in pixel
    """
    # H, W = radius_map.shape
    pad = ksize // 2
    padded = F.pad(
        radius_map.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect"
    )

    radii = []
    for x, y in centers:
        patch = padded[..., y : y + ksize, x : x + ksize]
        radii.append(patch.mean())  # 3×3 mean
    return torch.stack(radii)


def draw_mask(centers, radii, H, W):
    """
    centers: (K, 2)  (x, y)
    radii:   (K,)
    return:  (H, W) uint8 mask
    """
    radii *= 256
    mask = np.zeros((H, W), dtype=np.float32)
    for (x, y), r in zip(centers, radii):
        rr, cc = disk((y.item(), x.item()), r.item(), shape=mask.shape)
        mask[rr, cc] = 1
    return mask
