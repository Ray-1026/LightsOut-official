import os
import cv2
import glob
import random
import timeit
import numpy as np
import skimage
import yaml
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torch.distributions import Normal

# from utils.utils import RGB2YCbCr


class RandomGammaCorrection(object):
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, image):
        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0.5, 1, 2]
            self.gamma = random.choice(gammas)
            return TF.adjust_gamma(image, self.gamma, gain=1)
        elif isinstance(self.gamma, tuple):
            gamma = random.uniform(*self.gamma)
            return TF.adjust_gamma(image, gamma, gain=1)
        elif self.gamma == 0:
            return image
        else:
            return TF.adjust_gamma(image, self.gamma, gain=1)


def remove_background(image):
    # the input of the image is PIL.Image form with [H,W,C]
    image = np.float32(np.array(image))
    _EPS = 1e-7
    rgb_max = np.max(image, (0, 1))
    rgb_min = np.min(image, (0, 1))
    image = (image - rgb_min) * rgb_max / (rgb_max - rgb_min + _EPS)
    image = torch.from_numpy(image)
    return image


def glod_from_folder(folder_list, index_list):
    ext = ["png", "jpeg", "jpg", "bmp", "tif"]
    index_dict = {}
    for i, folder_name in enumerate(folder_list):
        data_list = []
        [data_list.extend(glob.glob(folder_name + "/*." + e)) for e in ext]
        data_list.sort()
        index_dict[index_list[i]] = data_list
    return index_dict


class Flare_Image_Loader(Dataset):
    def __init__(self, image_path, transform_base, transform_flare, mask_type=None):
        self.ext = ["png", "jpeg", "jpg", "bmp", "tif"]
        self.data_list = []
        [self.data_list.extend(glob.glob(image_path + "/*." + e)) for e in self.ext]
        self.flare_dict = {}
        self.flare_list = []
        self.flare_name_list = []

        self.reflective_flag = False
        self.reflective_dict = {}
        self.reflective_list = []
        self.reflective_name_list = []

        self.light_flag = False
        self.light_dict = {}
        self.light_list = []
        self.light_name_list = []

        self.mask_type = (
            mask_type  # It is a str which may be None,"luminance" or "color"
        )

        self.img_size = transform_base["img_size"]

        self.transform_base = transforms.Compose(
            [
                transforms.RandomCrop(
                    (self.img_size, self.img_size),
                    pad_if_needed=True,
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
            ]
        )

        self.transform_flare = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(0, 360),
                    scale=(transform_flare["scale_min"], transform_flare["scale_max"]),
                    translate=(
                        transform_flare["translate"] / 1440,
                        transform_flare["translate"] / 1440,
                    ),
                    shear=(-transform_flare["shear"], transform_flare["shear"]),
                ),
                transforms.CenterCrop((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

        self.normalize = transforms.Compose(
            [
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.data_ratio = []

    def lightsource_crop(self, matrix):
        """Find the largest rectangle of 1s in a binary matrix."""

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
        max_rectangle = [0, 0, 0, 0]  # (left, right, top, bottom)
        heights = torch.zeros(matrix.shape[1])

        for row in range(matrix.shape[0]):
            temp = 1 - matrix[row]
            heights = (heights + temp) * temp

            area, left, right, height = largestRectangleArea(heights.tolist())
            if area > max_area:
                max_area = area
                max_rectangle = [int(left), int(right), int(row - height + 1), int(row)]

        return torch.tensor(max_rectangle)

    def __getitem__(self, index):
        # load base image
        img_path = self.data_list[index]
        base_img = Image.open(img_path).convert("RGB")

        gamma = np.random.uniform(1.8, 2.2)
        to_tensor = transforms.ToTensor()
        adjust_gamma = RandomGammaCorrection(gamma)
        adjust_gamma_reverse = RandomGammaCorrection(1 / gamma)
        color_jitter = transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)
        if self.transform_base is not None:
            base_img = to_tensor(base_img)
            base_img = adjust_gamma(base_img)
            base_img = self.transform_base(base_img)
        else:
            base_img = to_tensor(base_img)
            base_img = adjust_gamma(base_img)
        sigma_chi = 0.01 * np.random.chisquare(df=1)
        base_img = Normal(base_img, sigma_chi).sample()
        gain = np.random.uniform(0.5, 1.2)
        flare_DC_offset = np.random.uniform(-0.02, 0.02)
        base_img = gain * base_img
        base_img = torch.clamp(base_img, min=0, max=1)

        choice_dataset = random.choices(
            [i for i in range(len(self.flare_list))], self.data_ratio
        )[0]
        choice_index = random.randint(0, len(self.flare_list[choice_dataset]) - 1)

        # load flare and light source image
        if self.light_flag:
            assert len(self.flare_list) == len(
                self.light_list
            ), "Error, number of light source and flares dataset no match!"
            for i in range(len(self.flare_list)):
                assert len(self.flare_list[i]) == len(
                    self.light_list[i]
                ), f"Error, number of light source and flares no match in {i} dataset!"
            flare_path = self.flare_list[choice_dataset][choice_index]
            light_path = self.light_list[choice_dataset][choice_index]
            light_img = Image.open(light_path).convert("RGB")
            light_img = to_tensor(light_img)
            light_img = adjust_gamma(light_img)
        else:
            flare_path = self.flare_list[choice_dataset][choice_index]
        flare_img = Image.open(flare_path).convert("RGB")
        if self.reflective_flag:
            reflective_path_list = self.reflective_list[choice_dataset]
            if len(reflective_path_list) != 0:
                reflective_path = random.choice(reflective_path_list)
                reflective_img = Image.open(reflective_path).convert("RGB")
            else:
                reflective_img = None

        flare_img = to_tensor(flare_img)
        flare_img = adjust_gamma(flare_img)

        if self.reflective_flag and reflective_img is not None:
            reflective_img = to_tensor(reflective_img)
            reflective_img = adjust_gamma(reflective_img)
            flare_img = torch.clamp(flare_img + reflective_img, min=0, max=1)

        flare_img = remove_background(flare_img)

        if self.transform_flare is not None:
            if self.light_flag:
                flare_merge = torch.cat((flare_img, light_img), dim=0)
                flare_merge = self.transform_flare(flare_merge)
            else:
                flare_img = self.transform_flare(flare_img)

        # change color
        if self.light_flag:
            # flare_merge=color_jitter(flare_merge)
            flare_img, light_img = torch.split(flare_merge, 3, dim=0)
        else:
            flare_img = color_jitter(flare_img)

        # flare blur
        blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))
        flare_img = blur_transform(flare_img)
        # flare_img=flare_img+flare_DC_offset
        flare_img = torch.clamp(flare_img, min=0, max=1)

        # merge image
        merge_img = flare_img + base_img
        merge_img = torch.clamp(merge_img, min=0, max=1)
        if self.light_flag:
            base_img = base_img + light_img
            base_img = torch.clamp(base_img, min=0, max=1)
            flare_img = flare_img - light_img
            flare_img = torch.clamp(flare_img, min=0, max=1)

        flare_mask = None
        if self.mask_type == None:
            return {
                "gt": adjust_gamma_reverse(base_img),
                "flare": adjust_gamma_reverse(flare_img),
                "lq": adjust_gamma_reverse(merge_img),
                "gamma": gamma,
            }

        elif self.mask_type == "luminance":
            # calculate mask (the mask is 3 channel)
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            luminance = 0.3 * flare_img[0] + 0.59 * flare_img[1] + 0.11 * flare_img[2]
            threshold_value = 0.99**gamma
            flare_mask = torch.where(luminance > threshold_value, one, zero)

        elif self.mask_type == "color":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            threshold_value = 0.99**gamma
            flare_mask = torch.where(merge_img > threshold_value, one, zero)

        elif self.mask_type == "flare":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            threshold_value = 0.7**gamma
            flare_mask = torch.where(flare_img > threshold_value, one, zero)

        elif self.mask_type == "light":
            # Depreciated: we dont need light mask anymore
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            luminance = 0.3 * light_img[0] + 0.59 * light_img[1] + 0.11 * light_img[2]
            threshold_value = 0.01
            flare_mask = torch.where(luminance > threshold_value, one, zero)

            light_source_cond = torch.zeros_like(flare_mask[0])
            light_source_cond = (flare_mask[0] + flare_mask[1] + flare_mask[2]) > 0
            light_source_cond = light_source_cond.float()
            light_source_cond = torch.repeat_interleave(
                light_source_cond[None, ...], 3, dim=0
            )

            # box = self.crop(light_source_cond[0])
            box = self.lightsource_crop(light_source_cond[0])

            # random int between -15 ~ 15
            margin = random.randint(-15, 15)

            if box[0] - margin >= 0:
                box[0] -= margin
            if box[1] + margin < self.img_size:
                box[1] += margin
            if box[2] - margin >= 0:
                box[2] -= margin
            if box[3] + margin < self.img_size:
                box[3] += margin

            top, bottom, left, right = box[2], box[3], box[0], box[1]

            merge_img = adjust_gamma_reverse(merge_img)

            cropped_mask = torch.ones((self.img_size, self.img_size))
            cropped_mask[top : bottom + 1, left : right + 1] = False
            cropped_mask = torch.repeat_interleave(cropped_mask[None, ...], 1, dim=0)

            channel3_mask = cropped_mask.repeat(3, 1, 1)
            masked_img = merge_img * (1 - channel3_mask)
            masked_img[channel3_mask == 1] = 0.5

        return {
            # add
            "pixel_values": self.normalize(merge_img),
            "masks": cropped_mask,
            "masked_images": self.normalize(masked_img),
            "conditioning_pixel_values": light_source_cond,
        }

    def __len__(self):
        return len(self.data_list)

    def load_scattering_flare(self, flare_name, flare_path):
        flare_list = []
        [flare_list.extend(glob.glob(flare_path + "/*." + e)) for e in self.ext]
        flare_list = sorted(flare_list)
        self.flare_name_list.append(flare_name)
        self.flare_dict[flare_name] = flare_list
        self.flare_list.append(flare_list)
        len_flare_list = len(self.flare_dict[flare_name])
        if len_flare_list == 0:
            print("ERROR: scattering flare images are not loaded properly")
        else:
            print(
                "Scattering Flare Image:",
                flare_name,
                " is loaded successfully with examples",
                str(len_flare_list),
            )
        # print("Now we have", len(self.flare_list), "scattering flare images")

    def load_light_source(self, light_name, light_path):
        # The number of the light source images should match the number of scattering flares
        light_list = []
        [light_list.extend(glob.glob(light_path + "/*." + e)) for e in self.ext]
        light_list = sorted(light_list)
        self.flare_name_list.append(light_name)
        self.light_dict[light_name] = light_list
        self.light_list.append(light_list)
        len_light_list = len(self.light_dict[light_name])

        if len_light_list == 0:
            print("ERROR: Light Source images are not loaded properly")
        else:
            self.light_flag = True
            print(
                "Light Source Image:",
                light_name,
                " is loaded successfully with examples",
                str(len_light_list),
            )
        # print("Now we have", len(self.light_list), "light source images")

    def load_reflective_flare(self, reflective_name, reflective_path):
        if reflective_path is None:
            reflective_list = []
        else:
            reflective_list = []
            [
                reflective_list.extend(glob.glob(reflective_path + "/*." + e))
                for e in self.ext
            ]
            reflective_list = sorted(reflective_list)
        self.reflective_name_list.append(reflective_name)
        self.reflective_dict[reflective_name] = reflective_list
        self.reflective_list.append(reflective_list)
        len_reflective_list = len(self.reflective_dict[reflective_name])
        if len_reflective_list == 0 and reflective_path is not None:
            print("ERROR: reflective flare images are not loaded properly")
        else:
            self.reflective_flag = True
            print(
                "Reflective Flare Image:",
                reflective_name,
                " is loaded successfully with examples",
                str(len_reflective_list),
            )
        # print("Now we have", len(self.reflective_list), "refelctive flare images")


class Flare7kpp_Pair_Loader(Flare_Image_Loader):
    def __init__(self, config):
        Flare_Image_Loader.__init__(
            self,
            config["image_path"],
            config["transform_base"],
            config["transform_flare"],
            config["mask_type"],
        )
        scattering_dict = config["scattering_dict"]
        reflective_dict = config["reflective_dict"]
        light_dict = config["light_dict"]

        # defualt not use light mask if opt['use_light_mask'] is not declared
        if "data_ratio" not in config or len(config["data_ratio"]) == 0:
            self.data_ratio = [1] * len(scattering_dict)
        else:
            self.data_ratio = config["data_ratio"]

        if len(scattering_dict) != 0:
            for key in scattering_dict.keys():
                self.load_scattering_flare(key, scattering_dict[key])
        if len(reflective_dict) != 0:
            for key in reflective_dict.keys():
                self.load_reflective_flare(key, reflective_dict[key])
        if len(light_dict) != 0:
            for key in light_dict.keys():
                self.load_light_source(key, light_dict[key])


class Lightsource_Regress_Loader(Flare7kpp_Pair_Loader):
    def __init__(self, config, num_lights=4):
        Flare7kpp_Pair_Loader.__init__(self, config)
        self.transform_flare = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(0, 360),
                    scale=(
                        config["transform_flare"]["scale_min"],
                        config["transform_flare"]["scale_max"],
                    ),
                    shear=(
                        -config["transform_flare"]["shear"],
                        config["transform_flare"]["shear"],
                    ),
                ),
                # transforms.CenterCrop((self.img_size, self.img_size)),
            ]
        )

        self.mask_type = "light"
        self.num_lights = num_lights

    def __getitem__(self, index):
        # load base image
        img_path = self.data_list[index]
        base_img = Image.open(img_path).convert("RGB")

        gamma = np.random.uniform(1.8, 2.2)
        to_tensor = transforms.ToTensor()
        adjust_gamma = RandomGammaCorrection(gamma)
        adjust_gamma_reverse = RandomGammaCorrection(1 / gamma)
        color_jitter = transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)

        base_img = to_tensor(base_img)
        base_img = adjust_gamma(base_img)
        if self.transform_base is not None:
            base_img = self.transform_base(base_img)

        sigma_chi = 0.01 * np.random.chisquare(df=1)
        base_img = Normal(base_img, sigma_chi).sample()
        gain = np.random.uniform(0.5, 1.2)
        base_img = gain * base_img
        base_img = torch.clamp(base_img, min=0, max=1)

        # init flare and light imgs
        flare_imgs = []
        light_imgs = []
        position = [
            [[-224, 0], [-224, 0]],
            [[-224, 0], [0, 224]],
            [[0, 224], [-224, 0]],
            [[0, 224], [0, 224]],
        ]
        axis = random.sample(range(4), 4)
        axis[-1] = axis[0]
        flare_nums = int(
            random.random() * self.num_lights + 1
        )  # random number of flares from 1 to 4

        for fn in range(flare_nums):
            choice_dataset = random.choices(
                [i for i in range(len(self.flare_list))], self.data_ratio
            )[0]
            choice_index = random.randint(0, len(self.flare_list[choice_dataset]) - 1)

            flare_path = self.flare_list[choice_dataset][choice_index]
            flare_img = Image.open(flare_path).convert("RGB")
            flare_img = to_tensor(flare_img)
            flare_img = adjust_gamma(flare_img)
            flare_img = remove_background(flare_img)

            if self.light_flag:
                light_path = self.light_list[choice_dataset][choice_index]
                light_img = Image.open(light_path).convert("RGB")
                light_img = to_tensor(light_img)
                light_img = adjust_gamma(light_img)

            if self.transform_flare is not None:
                if self.light_flag:
                    flare_merge = torch.cat((flare_img, light_img), dim=0)

                    if flare_nums == 1:
                        dx = random.randint(-224, 224)
                        dy = random.randint(-224, 224)
                    else:
                        dx = random.randint(
                            position[axis[fn]][0][0], position[axis[fn]][0][1]
                        )
                        dy = random.randint(
                            position[axis[fn]][1][0], position[axis[fn]][1][1]
                        )
                        if -160 < dx < 160 and -160 < dy < 160:
                            if random.random() < 0.5:
                                dx = 160 if dx > 0 else -160
                            else:
                                dy = 160 if dy > 0 else -160

                    flare_merge = self.transform_flare(flare_merge)
                    flare_merge = TF.affine(
                        flare_merge, angle=0, translate=(dx, dy), scale=1.0, shear=0
                    )
                    flare_merge = TF.center_crop(
                        flare_merge, (self.img_size, self.img_size)
                    )
                else:
                    flare_img = self.transform_flare(flare_img)

            # change color
            if self.light_flag:
                flare_img, light_img = torch.split(flare_merge, 3, dim=0)
            else:
                flare_img = color_jitter(flare_img)

            flare_imgs.append(flare_img)
            if self.light_flag:
                light_img = torch.clamp(light_img, min=0, max=1)
                light_imgs.append(light_img)

        flare_img = torch.sum(torch.stack(flare_imgs), dim=0)
        flare_img = torch.clamp(flare_img, min=0, max=1)

        # flare blur
        blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))
        flare_img = blur_transform(flare_img)
        flare_img = torch.clamp(flare_img, min=0, max=1)

        merge_img = torch.clamp(flare_img + base_img, min=0, max=1)

        if self.light_flag:
            light_img = torch.sum(torch.stack(light_imgs), dim=0)
            light_img = torch.clamp(light_img, min=0, max=1)
            base_img = torch.clamp(base_img + light_img, min=0, max=1)
            flare_img = torch.clamp(flare_img - light_img, min=0, max=1)

        flare_mask = None
        if self.mask_type == None:
            return {
                "gt": adjust_gamma_reverse(base_img),
                "flare": adjust_gamma_reverse(flare_img),
                "lq": adjust_gamma_reverse(merge_img),
                "gamma": gamma,
            }

        elif self.mask_type == "light":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)
            threshold_value = 0.01

            # flare_masks_list = []
            XYRs = torch.zeros((self.num_lights, 4))
            for i in range(flare_nums):
                luminance = (
                    0.3 * light_imgs[i][0]
                    + 0.59 * light_imgs[i][1]
                    + 0.11 * light_imgs[i][2]
                )
                flare_mask = torch.where(luminance > threshold_value, one, zero)

                light_source_cond = (flare_mask.sum(dim=0) > 0).float()

                x, y, r = self.find_circle_properties(light_source_cond, i)
                XYRs[i] = torch.tensor([x, y, r, 1.0])

            XYRs[:, :3] = XYRs[:, :3] / self.img_size

            luminance = 0.3 * light_img[0] + 0.59 * light_img[1] + 0.11 * light_img[2]
            flare_mask = torch.where(luminance > threshold_value, one, zero)

            light_source_cond = (flare_mask.sum(dim=0) > 0).float()

            light_source_cond = torch.repeat_interleave(
                light_source_cond[None, ...], 1, dim=0
            )

            # box = self.crop(light_source_cond[0])
            box = self.lightsource_crop(light_source_cond[0])

            # random int between 0 ~ 15
            margin = random.randint(0, 15)
            if box[0] - margin >= 0:
                box[0] -= margin
            if box[1] + margin < self.img_size:
                box[1] += margin
            if box[2] - margin >= 0:
                box[2] -= margin
            if box[3] + margin < self.img_size:
                box[3] += margin

            top, bottom, left, right = box[2], box[3], box[0], box[1]

            merge_img = adjust_gamma_reverse(merge_img)

            cropped_mask = torch.full(
                (self.img_size, self.img_size), True, dtype=torch.bool
            )
            cropped_mask[top : bottom + 1, left : right + 1] = False
            channel3_mask = cropped_mask.unsqueeze(0).expand(3, -1, -1)

            masked_img = merge_img * (1 - channel3_mask.float())
            masked_img[channel3_mask] = 0.5

        return {
            # add
            "input": self.normalize(masked_img),  # normalize to [-1, 1]
            "light_masks": light_source_cond,
            "xyrs": XYRs,
        }

    def find_circle_properties(self, mask, i, method="minEnclosingCircle"):
        """
        Find the properties of the light source circle in the mask.
        """

        _mask = (mask.numpy() * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return 0.0, 0.0, 0.0

        largest_contour = max(contours, key=cv2.contourArea)

        if method == "minEnclosingCircle":
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        elif method == "area_based":
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:  # if the contour is too small
                return 0.0, 0.0, 0.0

            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
            area = cv2.contourArea(largest_contour)
            radius = np.sqrt(area / np.pi)

        # # draw
        # cv2.circle(_mask, (int(x), int(y)), int(radius), 128, 2)
        # cv2.imwrite(f"mask_{i}.png", _mask)

        return x, y, radius


class Lightsource_3Maps_Loader(Lightsource_Regress_Loader):
    def __init__(self, config, num_lights=4):
        Lightsource_Regress_Loader.__init__(self, config, num_lights=num_lights)

    def build_gt_maps(self, coords, radii, H, W, kappa=0.4):
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        prob_gt = torch.zeros((H, W))
        rad_gt = torch.zeros((H, W))

        eps = 1e-6
        for x_i, y_i, r_i in zip(coords[:, 0], coords[:, 1], radii):
            if r_i < 1.0:
                continue

            sigma = kappa * r_i
            g = torch.exp(-((xx - x_i) ** 2 + (yy - y_i) ** 2) / (2 * sigma**2))
            g_prime = torch.exp(
                -((xx - x_i) ** 2 + (yy - y_i) ** 2) / (2 * (sigma / 1.414) ** 2)
            )
            prob_gt = torch.maximum(prob_gt, g)
            rad_gt = torch.maximum(rad_gt, g_prime * r_i)

        rad_gt = rad_gt / (prob_gt + eps)
        return prob_gt, rad_gt

    def __getitem__(self, index):
        # load base image
        img_path = self.data_list[index]
        base_img = Image.open(img_path).convert("RGB")

        gamma = np.random.uniform(1.8, 2.2)
        to_tensor = transforms.ToTensor()
        adjust_gamma = RandomGammaCorrection(gamma)
        adjust_gamma_reverse = RandomGammaCorrection(1 / gamma)
        color_jitter = transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)

        base_img = to_tensor(base_img)
        base_img = adjust_gamma(base_img)
        if self.transform_base is not None:
            base_img = self.transform_base(base_img)

        sigma_chi = 0.01 * np.random.chisquare(df=1)
        base_img = Normal(base_img, sigma_chi).sample()
        gain = np.random.uniform(0.5, 1.2)
        base_img = gain * base_img
        base_img = torch.clamp(base_img, min=0, max=1)

        # init flare and light imgs
        flare_imgs = []
        light_imgs = []
        position = [
            [[-224, 0], [-224, 0]],
            [[-224, 0], [0, 224]],
            [[0, 224], [-224, 0]],
            [[0, 224], [0, 224]],
        ]
        axis = random.sample(range(4), 4)
        axis[-1] = axis[0]
        flare_nums = int(
            random.random() * self.num_lights + 1
        )  # random number of flares from 1 to 4

        for fn in range(flare_nums):
            choice_dataset = random.choices(
                [i for i in range(len(self.flare_list))], self.data_ratio
            )[0]
            choice_index = random.randint(0, len(self.flare_list[choice_dataset]) - 1)

            flare_path = self.flare_list[choice_dataset][choice_index]
            flare_img = Image.open(flare_path).convert("RGB")
            flare_img = to_tensor(flare_img)
            flare_img = adjust_gamma(flare_img)
            flare_img = remove_background(flare_img)

            if self.light_flag:
                light_path = self.light_list[choice_dataset][choice_index]
                light_img = Image.open(light_path).convert("RGB")
                light_img = to_tensor(light_img)
                light_img = adjust_gamma(light_img)

            if self.transform_flare is not None:
                if self.light_flag:
                    flare_merge = torch.cat((flare_img, light_img), dim=0)

                    if flare_nums == 1:
                        dx = random.randint(-224, 224)
                        dy = random.randint(-224, 224)
                    else:
                        dx = random.randint(
                            position[axis[fn]][0][0], position[axis[fn]][0][1]
                        )
                        dy = random.randint(
                            position[axis[fn]][1][0], position[axis[fn]][1][1]
                        )
                        if -160 < dx < 160 and -160 < dy < 160:
                            if random.random() < 0.5:
                                dx = 160 if dx > 0 else -160
                            else:
                                dy = 160 if dy > 0 else -160

                    flare_merge = self.transform_flare(flare_merge)
                    flare_merge = TF.affine(
                        flare_merge, angle=0, translate=(dx, dy), scale=1.0, shear=0
                    )
                    flare_merge = TF.center_crop(
                        flare_merge, (self.img_size, self.img_size)
                    )
                else:
                    flare_img = self.transform_flare(flare_img)

            # change color
            if self.light_flag:
                flare_img, light_img = torch.split(flare_merge, 3, dim=0)
            else:
                flare_img = color_jitter(flare_img)

            flare_imgs.append(flare_img)
            if self.light_flag:
                light_img = torch.clamp(light_img, min=0, max=1)
                light_imgs.append(light_img)

        flare_img = torch.sum(torch.stack(flare_imgs), dim=0)
        flare_img = torch.clamp(flare_img, min=0, max=1)

        # flare blur
        blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))
        flare_img = blur_transform(flare_img)
        flare_img = torch.clamp(flare_img, min=0, max=1)

        merge_img = torch.clamp(flare_img + base_img, min=0, max=1)

        if self.light_flag:
            light_img = torch.sum(torch.stack(light_imgs), dim=0)
            light_img = torch.clamp(light_img, min=0, max=1)
            base_img = torch.clamp(base_img + light_img, min=0, max=1)
            flare_img = torch.clamp(flare_img - light_img, min=0, max=1)

        flare_mask = None
        if self.mask_type == None:
            return {
                "gt": adjust_gamma_reverse(base_img),
                "flare": adjust_gamma_reverse(flare_img),
                "lq": adjust_gamma_reverse(merge_img),
                "gamma": gamma,
            }

        elif self.mask_type == "light":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)
            threshold_value = 0.01

            # flare_masks_list = []
            XYRs = torch.zeros((self.num_lights, 4))
            for i in range(flare_nums):
                luminance = (
                    0.3 * light_imgs[i][0]
                    + 0.59 * light_imgs[i][1]
                    + 0.11 * light_imgs[i][2]
                )
                flare_mask = torch.where(luminance > threshold_value, one, zero)

                light_source_cond = (flare_mask.sum(dim=0) > 0).float()

                x, y, r = self.find_circle_properties(light_source_cond, i)
                XYRs[i] = torch.tensor([x, y, r, 1.0])

            gt_prob, gt_rad = self.build_gt_maps(
                XYRs[:, :2], XYRs[:, 2], self.img_size, self.img_size
            )
            gt_prob = gt_prob.unsqueeze(0)  # shape: (1, H, W)
            gt_rad = gt_rad.unsqueeze(0)
            gt_rad /= self.img_size
            gt_maps = torch.cat((gt_prob, gt_rad), dim=0)  # shape: (2, H, W)

            XYRs[:, :3] = XYRs[:, :3] / self.img_size

            luminance = 0.3 * light_img[0] + 0.59 * light_img[1] + 0.11 * light_img[2]
            flare_mask = torch.where(luminance > threshold_value, one, zero)

            light_source_cond = (flare_mask.sum(dim=0) > 0).float()

            light_source_cond = torch.repeat_interleave(
                light_source_cond[None, ...], 1, dim=0
            )

            # box = self.crop(light_source_cond[0])
            box = self.lightsource_crop(light_source_cond[0])

            # random int between 0 ~ 15
            margin = random.randint(0, 15)
            if box[0] - margin >= 0:
                box[0] -= margin
            if box[1] + margin < self.img_size:
                box[1] += margin
            if box[2] - margin >= 0:
                box[2] -= margin
            if box[3] + margin < self.img_size:
                box[3] += margin

            top, bottom, left, right = box[2], box[3], box[0], box[1]

            merge_img = adjust_gamma_reverse(merge_img)

            cropped_mask = torch.full(
                (self.img_size, self.img_size), True, dtype=torch.bool
            )
            cropped_mask[top : bottom + 1, left : right + 1] = False
            channel3_mask = cropped_mask.unsqueeze(0).expand(3, -1, -1)

            masked_img = merge_img * (1 - channel3_mask.float())
            masked_img[channel3_mask] = 0.5

            return {
                # add
                "input": self.normalize(masked_img),  # normalize to [-1, 1]
                "light_masks": light_source_cond,
                "xyrs": gt_maps,
            }


class TestImageLoader(Dataset):
    def __init__(
        self,
        dataroot_gt,
        dataroot_input,
        dataroot_mask,
        margin=0,
        img_size=512,
        noise_matching=False,
    ):
        super(TestImageLoader, self).__init__()
        self.gt_folder = dataroot_gt
        self.input_folder = dataroot_input
        self.mask_folder = dataroot_mask
        self.paths = glod_from_folder(
            [self.input_folder, self.gt_folder, self.mask_folder],
            ["input", "gt", "mask"],
        )

        self.margin = margin
        self.img_size = img_size
        self.noise_matching = noise_matching

    def __len__(self):
        return len(self.paths["input"])

    def __getitem__(self, index):
        img_name = self.paths["input"][index].split("/")[-1]
        num = img_name.split("_")[1].split(".")[0]

        # preprocess light source mask
        light_mask = np.array(Image.open(self.paths["mask"][index]))
        tmp_light_mask = np.zeros_like(light_mask[:, :, 0])
        tmp_light_mask[light_mask[:, :, 2] > 0] = 255
        cond = (light_mask[:, :, 0] > 0) & (light_mask[:, :, 1] > 0)
        tmp_light_mask[cond] = 0
        light_mask = tmp_light_mask

        # img for controlnet input
        control_img = np.repeat(light_mask[:, :, None], 3, axis=2)

        # crop region
        box = self.lightsource_crop(light_mask)

        if box[0] - self.margin >= 0:
            box[0] -= self.margin
        if box[1] + self.margin < self.img_size:
            box[1] += self.margin
        if box[2] - self.margin >= 0:
            box[2] -= self.margin
        if box[3] + self.margin < self.img_size:
            box[3] += self.margin

        # input image to be outpainted
        input_img = np.array(Image.open(self.paths["input"][index]))
        cropped_region = np.ones((self.img_size, self.img_size), dtype=np.uint8)
        cropped_region[box[2] : box[3] + 1, box[0] : box[1] + 1] = 0
        input_img[cropped_region == 1] = 128

        # image for blip
        blip_img = input_img[box[2] : box[3] + 1, box[0] : box[1] + 1, :]

        # noise matching
        input_img_matching = None
        if self.noise_matching:
            np_src_img = input_img / 255.0
            np_mask_rgb = np.repeat(cropped_region[:, :, None], 3, axis=2).astype(
                np.float32
            )
            matched_noise = self.get_matched_noise(np_src_img, np_mask_rgb)
            input_img_matching = (matched_noise * 255).astype(np.uint8)

        # mask image
        mask_img = (cropped_region * 255).astype(np.uint8)

        return {
            "blip_img": blip_img,
            "input_img": Image.fromarray(input_img),
            "input_img_matching": (
                Image.fromarray(input_img_matching)
                if input_img_matching is not None
                else Image.fromarray(input_img)
            ),
            "mask_img": Image.fromarray(mask_img),
            "control_img": Image.fromarray(control_img),
            "box": box,
            "output_name": "output_" + num + ".png",
        }

    def lightsource_crop(self, matrix):
        """Find the largest rectangle of 1s in a binary matrix."""

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
        max_rectangle = [0, 0, 0, 0]  # (left, right, top, bottom)
        heights = [0] * len(matrix[0])
        for row in range(len(matrix)):
            for i, val in enumerate(matrix[row]):
                heights[i] = heights[i] + 1 if val == 0 else 0

            area, left, right, height = largestRectangleArea(heights)
            if area > max_area:
                max_area = area
                max_rectangle = [int(left), int(right), int(row - height + 1), int(row)]

        return list(max_rectangle)

    # this function is taken from https://github.com/parlance-zz/g-diffuser-bot
    def get_matched_noise(
        self, _np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05
    ):
        # helper fft routines that keep ortho normalization and auto-shift before and after fft
        def _fft2(data):
            if data.ndim > 2:  # has channels
                out_fft = np.zeros(
                    (data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128
                )
                for c in range(data.shape[2]):
                    c_data = data[:, :, c]
                    out_fft[:, :, c] = np.fft.fft2(
                        np.fft.fftshift(c_data), norm="ortho"
                    )
                    out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
            else:  # one channel
                out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
                out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
                out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

            return out_fft

        def _ifft2(data):
            if data.ndim > 2:  # has channels
                out_ifft = np.zeros(
                    (data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128
                )
                for c in range(data.shape[2]):
                    c_data = data[:, :, c]
                    out_ifft[:, :, c] = np.fft.ifft2(
                        np.fft.fftshift(c_data), norm="ortho"
                    )
                    out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
            else:  # one channel
                out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
                out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
                out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

            return out_ifft

        def _get_gaussian_window(width, height, std=3.14, mode=0):
            window_scale_x = float(width / min(width, height))
            window_scale_y = float(height / min(width, height))

            window = np.zeros((width, height))
            x = (np.arange(width) / width * 2.0 - 1.0) * window_scale_x
            for y in range(height):
                fy = (y / height * 2.0 - 1.0) * window_scale_y
                if mode == 0:
                    window[:, y] = np.exp(-(x**2 + fy**2) * std)
                else:
                    window[:, y] = (1 / ((x**2 + 1.0) * (fy**2 + 1.0))) ** (
                        std / 3.14
                    )  # hey wait a minute that's not gaussian

            return window

        def _get_masked_window_rgb(np_mask_grey, hardness=1.0):
            np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
            if hardness != 1.0:
                hardened = np_mask_grey[:] ** hardness
            else:
                hardened = np_mask_grey[:]
            for c in range(3):
                np_mask_rgb[:, :, c] = hardened[:]
            return np_mask_rgb

        width = _np_src_image.shape[0]
        height = _np_src_image.shape[1]
        num_channels = _np_src_image.shape[2]

        _np_src_image[:] * (1.0 - np_mask_rgb)
        np_mask_grey = np.sum(np_mask_rgb, axis=2) / 3.0
        img_mask = np_mask_grey > 1e-6
        ref_mask = np_mask_grey < 1e-3

        windowed_image = _np_src_image * (1.0 - _get_masked_window_rgb(np_mask_grey))
        windowed_image /= np.max(windowed_image)
        windowed_image += (
            np.average(_np_src_image) * np_mask_rgb
        )  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color

        src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
        src_dist = np.absolute(src_fft)
        src_phase = src_fft / src_dist

        # create a generator with a static seed to make outpainting deterministic / only follow global seed
        rng = np.random.default_rng(0)

        noise_window = _get_gaussian_window(
            width, height, mode=1
        )  # start with simple gaussian noise
        noise_rgb = rng.random((width, height, num_channels))
        noise_grey = np.sum(noise_rgb, axis=2) / 3.0
        noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
        for c in range(num_channels):
            noise_rgb[:, :, c] += (1.0 - color_variation) * noise_grey

        noise_fft = _fft2(noise_rgb)
        for c in range(num_channels):
            noise_fft[:, :, c] *= noise_window
        noise_rgb = np.real(_ifft2(noise_fft))
        shaped_noise_fft = _fft2(noise_rgb)
        shaped_noise_fft[:, :, :] = (
            np.absolute(shaped_noise_fft[:, :, :]) ** 2
            * (src_dist**noise_q)
            * src_phase
        )  # perform the actual shaping

        brightness_variation = 0.0  # color_variation # todo: temporarily tying brightness variation to color variation for now
        contrast_adjusted_np_src = (
            _np_src_image[:] * (brightness_variation + 1.0) - brightness_variation * 2.0
        )

        # scikit-image is used for histogram matching, very convenient!
        shaped_noise = np.real(_ifft2(shaped_noise_fft))
        shaped_noise -= np.min(shaped_noise)
        shaped_noise /= np.max(shaped_noise)
        shaped_noise[img_mask, :] = skimage.exposure.match_histograms(
            shaped_noise[img_mask, :] ** 1.0,
            contrast_adjusted_np_src[ref_mask, :],
            channel_axis=1,
        )
        shaped_noise = (
            _np_src_image[:] * (1.0 - np_mask_rgb) + shaped_noise * np_mask_rgb
        )

        matched_noise = shaped_noise[:]

        return np.clip(matched_noise, 0.0, 1.0)


if __name__ == "__main__":
    config_path = "configs/flare7kpp_dataset.yml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    dataset = TestImageLoader(
        config["testing_dataset"]["dataroot_gt"],
        config["testing_dataset"]["dataroot_lq"],
        config["testing_dataset"]["dataroot_mask"],
        margin=0,
    )

    for i, d in enumerate(dataset):
        if i == 5:
            break
