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
from skimage import morphology
from collections import OrderedDict


def load_mfdnet_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(0))
    new_state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module"):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict)


def adjust_gamma(image: torch.Tensor, gamma):
    # image is in shape of [B,C,H,W] and gamma is in shape [B]
    gamma = gamma.float().cuda()
    gamma_tensor = torch.ones_like(image)
    gamma_tensor = gamma.view(-1, 1, 1, 1) * gamma_tensor
    image = torch.pow(image, gamma_tensor)
    out = torch.clamp(image, 0.0, 1.0)
    return out


def adjust_gamma_reverse(image: torch.Tensor, gamma):
    # gamma=torch.Tensor([gamma]).cuda()
    gamma = 1 / gamma.float().cuda()
    gamma_tensor = torch.ones_like(image)
    gamma_tensor = gamma.view(-1, 1, 1, 1) * gamma_tensor
    image = torch.pow(image, gamma_tensor)
    out = torch.clamp(image, 0.0, 1.0)
    return out


def predict_flare_from_6_channel(input_tensor, gamma):
    # the input is a tensor in [B,C,H,W], the C here is 6

    deflare_img = input_tensor[:, :3, :, :]
    flare_img_predicted = input_tensor[:, 3:, :, :]

    merge_img_predicted_linear = adjust_gamma(deflare_img, gamma) + adjust_gamma(
        flare_img_predicted, gamma
    )
    merge_img_predicted = adjust_gamma_reverse(
        torch.clamp(merge_img_predicted_linear, 1e-7, 1.0), gamma
    )
    return deflare_img, flare_img_predicted, merge_img_predicted


def predict_flare_from_3_channel(
    input_tensor, flare_mask, base_img, flare_img, merge_img, gamma
):
    # the input is a tensor in [B,C,H,W], the C here is 3

    input_tensor_linear = adjust_gamma(input_tensor, gamma)
    merge_tensor_linear = adjust_gamma(merge_img, gamma)
    flare_img_predicted = adjust_gamma_reverse(
        torch.clamp(merge_tensor_linear - input_tensor_linear, 1e-7, 1.0), gamma
    )

    masked_deflare_img = input_tensor * (1 - flare_mask) + base_img * flare_mask
    masked_flare_img_predicted = (
        flare_img_predicted * (1 - flare_mask) + flare_img * flare_mask
    )

    return masked_deflare_img, masked_flare_img_predicted


def get_highlight_mask(image, threshold=0.99, luminance_mode=False):
    """Get the area close to the exposure
    Args:
        image: the image tensor in [B,C,H,W]. For inference, B is set as 1.
        threshold: the threshold of luminance/greyscale of exposure region
        luminance_mode: use luminance or greyscale
    Return:
        Binary image in [B,H,W]
    """
    if luminance_mode:
        # 3 channels in RGB
        luminance = (
            0.2126 * image[:, 0, :, :]
            + 0.7152 * image[:, 1, :, :]
            + 0.0722 * image[:, 2, :, :]
        )
        binary_mask = luminance > threshold
    else:
        binary_mask = image.mean(dim=1, keepdim=True) > threshold
    binary_mask = binary_mask.to(image.dtype)
    return binary_mask


def refine_mask(mask, morph_size=0.01):
    """Refines a mask by applying mophological operations.
    Args:
      mask: A float array of shape [H, W]
      morph_size: Size of the morphological kernel relative to the long side of
        the image.

    Returns:
      Refined mask of shape [H, W].
    """
    mask_size = max(np.shape(mask))
    kernel_radius = 0.5 * morph_size * mask_size
    kernel = morphology.disk(np.ceil(kernel_radius))
    opened = morphology.binary_opening(mask, kernel)
    return opened


def _create_disk_kernel(kernel_size):
    _EPS = 1e-7
    x = np.arange(kernel_size) - (kernel_size - 1) / 2
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2 + yy**2)
    kernel = np.float32(rr <= np.max(x)) + _EPS
    kernel = kernel / np.sum(kernel)
    return kernel


def blend_light_source(input_scene, pred_scene, threshold=0.99, luminance_mode=False):
    binary_mask = (
        get_highlight_mask(
            input_scene, threshold=threshold, luminance_mode=luminance_mode
        )
        > 0.5
    ).to("cpu", torch.bool)
    binary_mask = binary_mask.squeeze()  # (h, w)
    binary_mask = binary_mask.numpy()
    binary_mask = refine_mask(binary_mask)

    labeled = skimage.measure.label(binary_mask)
    properties = skimage.measure.regionprops(labeled)
    max_diameter = 0
    for p in properties:
        # The diameter of a circle with the same area as the region.
        max_diameter = max(max_diameter, p["equivalent_diameter"])

    mask = np.float32(binary_mask)
    kernel_size = round(1.5 * max_diameter)  # default is 1.5
    if kernel_size > 0:
        kernel = _create_disk_kernel(kernel_size)
        mask = cv2.filter2D(mask, -1, kernel)
        mask = np.clip(mask * 3.0, 0.0, 1.0)
        mask_rgb = np.stack([mask] * 3, axis=0)

        mask_rgb = torch.from_numpy(mask_rgb).to(input_scene.device, torch.float32)
        blend = input_scene * mask_rgb + pred_scene * (1 - mask_rgb)
    else:
        blend = pred_scene
    return blend


def blend_with_alpha(result, input_img, box, blur_size=31):
    """
    Apply alpha blending to paste the specified box region from input_img onto the result image
    to reduce boundary artifacts and make the blending more natural.

    Args:
        result (np.array): inpainting generated image
        input_img (np.array): original image
        box (tuple): (x_min, x_max, y_min, y_max) representing the paste-back region from the original image
        blur_size (int): blur range for the mask, larger values create smoother transitions (recommended 15~50)

    Returns:
        np.array: image after alpha blending
    """

    x_min, x_max, y_min, y_max = box

    # alpha mask
    mask = np.zeros_like(result, dtype=np.float32)
    mask[y_min : y_max + 1, x_min : x_max + 1] = 1.0

    # gaussian blur
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # alpha blending
    blended = (mask * input_img + (1 - mask) * result).astype(np.uint8)

    return blended


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
