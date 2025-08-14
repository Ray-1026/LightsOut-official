import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
from torchvision.transforms import ToTensor
import numpy as np
from glob import glob
import lpips
from tqdm import tqdm
import cv2
from PIL import Image
import random
import os

import warnings

warnings.filterwarnings("ignore")


def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
    return output_lpips.cpu().detach().numpy()[0, 0, 0, 0]


def compare_score(img1, img2, img_seg):
    # Return the G-PSNR, S-PSNR, Global-PSNR and Score
    # This module is for the MIPI 2023 Challange: https://codalab.lisn.upsaclay.fr/competitions/9402
    mask_type_list = ["glare", "streak", "global"]
    metric_dict = {"glare": 0, "streak": 0, "global": 0}
    for mask_type in mask_type_list:
        mask_area, img_mask = extract_mask(img_seg)[mask_type]
        if mask_area > 0:
            img_gt_masked = img1 * img_mask
            img_input_masked = img2 * img_mask
            input_mse = compare_mse(img_gt_masked, img_input_masked) / (
                255 * 255 * mask_area
            )
            input_psnr = 10 * np.log10((1.0**2) / input_mse)
            metric_dict[mask_type] = input_psnr
        else:
            metric_dict.pop(mask_type)
    return metric_dict


def extract_mask(img_seg):
    # Return a dict with 3 masks including streak,glare,global(whole image w/o light source), masks are returned in 3ch.
    # glare: [255,255,0]
    # streak: [255,0,0]
    # light source: [0,0,255]
    # others: [0,0,0]
    mask_dict = {}
    streak_mask = (img_seg[:, :, 0] - img_seg[:, :, 1]) / 255
    glare_mask = (img_seg[:, :, 1]) / 255
    global_mask = (255 - img_seg[:, :, 2]) / 255
    mask_dict["glare"] = [
        np.sum(glare_mask) / (512 * 512),
        np.expand_dims(glare_mask, 2).repeat(3, axis=2),
    ]  # area, mask
    mask_dict["streak"] = [
        np.sum(streak_mask) / (512 * 512),
        np.expand_dims(streak_mask, 2).repeat(3, axis=2),
    ]
    mask_dict["global"] = [
        np.sum(global_mask) / (512 * 512),
        np.expand_dims(global_mask, 2).repeat(3, axis=2),
    ]
    return mask_dict


def max_rectangle(matrix):
    def largestRectangleArea(heights):
        heights.append(0)
        stack = [-1]
        max_area = 0
        max_rectangle = (0, 0, 0, 0)  # (area, left, right, height)
        for i in range(len(heights)):
            while heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    max_rectangle = (area, stack[-1] + 1, i - 1, h)
            stack.append(i)
        heights.pop()
        return max_rectangle

    max_area = 0
    max_rectangle = (0, 0, 0, 0)  # (left, right, top, bottom)
    heights = [0] * len(matrix[0])
    for row in range(len(matrix)):
        for i, val in enumerate(matrix[row]):
            heights[i] = heights[i] + 1 if val == 0 else 0
        area, left, right, height = largestRectangleArea(heights)
        if area > max_area:
            max_area = area
            max_rectangle = (left, right, row - height + 1, row)

    return list(max_rectangle)


def cal_mask(mask):
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    new_mask[mask[:, :, 2] > 0] = 255
    cond = (mask[:, :, 0] > 0) & (mask[:, :, 1] > 0)
    new_mask[cond] = 0

    return new_mask


def calculate_metrics(args):
    loss_fn_alex = lpips.LPIPS(net="alex").cuda()
    gt_folder = args["gt"] + "/*"
    input_folder = args["input"] + "/*"
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))
    if args["mask"] is not None:
        mask_folder = args["mask"] + "/*"
        mask_list = sorted(glob(mask_folder))

    n = len(gt_list)

    # ssim, psnr, lpips_val = 0, 0, 0
    ssim, psnr, lpips_val = [], [], []
    g_psnr, s_psnr = [], []
    score_dict = {
        "glare": 0,
        "streak": 0,
        "global": 0,
        "glare_num": 0,
        "streak_num": 0,
        "global_num": 0,
    }
    for i in tqdm(range(n)):
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])
        mask = io.imread(mask_list[i])
        mask = cal_mask(mask)
        res = max_rectangle(mask)

        margin = args["crop_margin"]
        if res[0] - margin >= 0:
            res[0] -= margin
        if res[1] + margin < 512:
            res[1] += margin
        if res[2] - margin >= 0:
            res[2] -= margin
        if res[3] + margin < 512:
            res[3] += margin
        left, right, top, bottom = res[0], res[1], res[2], res[3]

        name = args["name"]

        img_gt = img_gt[top : bottom + 1, left : right + 1]

        if name != "baseline":  # baseline does not crop
            img_input = img_input[top : bottom + 1, left : right + 1]

        size = 512
        img_gt = np.array(
            Image.fromarray(img_gt).resize((size, size), Image.LANCZOS).convert("RGB")
        )
        img_input = np.array(
            Image.fromarray(img_input).resize((size, size), Image.LANCZOS)
        )

        if name != "baseline":  # only save cropped images for non-baseline methods
            os.makedirs(args["crop"], exist_ok=True)
            Image.fromarray(img_input).save(f"{args["crop"]}/{i:06d}.png")

        ssim += [compare_ssim(img_gt, img_input, multichannel=True, channel_axis=2)]
        psnr += [compare_psnr(img_gt, img_input, data_range=255)]
        lpips_val += [compare_lpips(img_gt, img_input, loss_fn_alex)]
        if args["mask"] is not None:
            img_seg = io.imread(mask_list[i])

            if name != "baseline":  # baseline does not crop
                img_seg = img_seg[top : bottom + 1, left : right + 1]

            img_seg = np.array(
                Image.fromarray(img_seg).resize((size, size), Image.LANCZOS)
            )
            metric_dict = compare_score(img_gt, img_input, img_seg)
            g_psnr += [metric_dict["glare"] if "glare" in metric_dict.keys() else 0]
            s_psnr += [metric_dict["streak"] if "streak" in metric_dict.keys() else 0]

            for key in metric_dict.keys():
                score_dict[key] += metric_dict[key]
                score_dict[key + "_num"] += 1

    glare_psnr, streak_psnr = -np.inf, -np.inf
    if args["mask"] is not None:
        for key in ["glare", "streak", "global"]:
            if score_dict[key + "_num"] == 0:
                assert False, "Error, No mask in this type!"
            score_dict[key] /= score_dict[key + "_num"]
        score_dict["score"] = (
            1 / 3 * (score_dict["glare"] + score_dict["global"] + score_dict["streak"])
        )
        glare_psnr = score_dict["glare"]
        streak_psnr = score_dict["streak"]

    print(
        f"PSNR: {np.mean(psnr):.2f}, SSIM: {np.mean(ssim):.4f}, LPIPS: {np.mean(lpips_val):.4f}, G-PSNR: {glare_psnr:.2f}, S-PSNR: {streak_psnr:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--crop", type=str, default=None)
    parser.add_argument("--crop_margin", type=int, default=0)
    parser.add_argument("--name", type=str, default="lightsout", choices=["lightsout", "baseline"])
    args = vars(parser.parse_args())
    calculate_metrics(args)
