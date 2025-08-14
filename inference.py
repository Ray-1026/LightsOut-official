import argparse
import numpy as np
import cv2
import os
import torch
import yaml
import torchvision
from PIL import Image, ImageFilter
from tqdm import tqdm
from diffusers import ControlNetModel, DPMSolverMultistepScheduler
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from src.pipelines.pipeline_stable_diffusion_outpaint import OutpaintPipeline
from src.pipelines.pipeline_controlnet_outpaint import ControlNetOutpaintPipeline
from src.schedulers.scheduling_pndm import CustomScheduler
from src.models.unet import U_Net
from src.models.light_source_regressor import LightSourceRegressor
from utils.dataset import TestImageLoader
from utils.utils import IoU, mean_IoU


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--sd_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--light_regress_path",
        type=str,
        default="pretrained/light_regress/model.pth",
        help="Path to pretrained light regress model.",
    )
    parser.add_argument(
        "--light_control_path",
        type=str,
        default="pretrained/light_control",
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--light_outpaint_path",
        type=str,
        default="pretrained/lora_sd",
        help="Path to lora model.",
    )
    parser.add_argument(
        "--blip2_path",
        type=str,
        default="Salesforce/blip2-opt-2.7b",
        help="Path to pretrained blip2 model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="res",
        help="The output directory where predictions are saved",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/flare7kpp_dataset.yml",
        help="Path to dataset configuration file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible inference."
    )
    parser.add_argument(
        "--light_setting",
        choices=["incomplete", "no"],
        default="no",
        help="Light setting for the model.",
    )
    parser.add_argument(
        "--step", type=int, default=50, help="Number of denoising steps."
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.5,
        help="CFG scale for the model.",
    )
    parser.add_argument(
        "--noise_reinjection",
        type=int,
        default=4,
        help="Number of noise reinjection steps.",
    )
    parser.add_argument(
        "--additional_prompt",
        type=str,
        default=None,
        help="Additional prompt for the model.",
    )

    args = parser.parse_args()

    # set margin
    if args.light_setting == "incomplete":
        args.margin = 15
    else:
        args.margin = 0

    return args


def LSRMInit(light_regress_model, device="cuda"):
    """light source regression module"""
    model = LightSourceRegressor()
    ckpt = torch.load(light_regress_model)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def OutpainterInit(
    sd_model, controlnet, lora, device="cuda", torch_dtype=torch.float16
):
    if controlnet is not None:
        control_net = ControlNetModel.from_pretrained(
            controlnet, torch_dtype=torch_dtype
        )
        pipe = ControlNetOutpaintPipeline.from_pretrained(
            sd_model, controlnet=control_net, torch_dtype=torch_dtype
        )
    else:
        pipe = OutpaintPipeline.from_pretrained(sd_model, torch_dtype=torch_dtype)

    # add my scheduler
    pipe.scheduler = CustomScheduler.from_config(pipe.scheduler.config)

    # add lora
    if lora is not None:
        pipe.unet.load_attn_procs(lora, use_safetensors=True)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def Blip2Init(blip2_path, device="cuda", torch_dtype=torch.float16):
    processor = AutoProcessor.from_pretrained(blip2_path)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(
        blip2_path, torch_dtype=torch_dtype
    )
    blip2 = blip2.to(device)
    return processor, blip2


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


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # lightsource regress model
    lsr_module = LSRMInit(args.light_regress_path)

    # sd outpaint
    pipe = OutpainterInit(args.sd_path, args.controlnet_path, args.light_outpaint_path)

    # blip2
    processor, blip2 = Blip2Init(args.blip2_path)

    # dataset
    with open(args.dataset_config, "r") as stream:
        config = yaml.safe_load(stream)

    dataset = TestImageLoader(
        config["testing_dataset"]["dataroot_gt"],
        config["testing_dataset"]["dataroot_lq"],
        config["testing_dataset"]["dataroot_mask"],
        margin=args.margin,  # 0 or 15
    )

    # generator
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # test transformation
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    threshold = 0.5
    # iou_scores = []

    for data in tqdm(dataset):
        with torch.no_grad():
            input_img = data["input_img"]
            control_img = data["control_img"]

            input_img = transform(input_img).unsqueeze(0).to("cuda")
            control_img = transform(control_img).unsqueeze(0)

            pred_mask = lsr_module.forward_render(input_img)

            pred_mask = (pred_mask > threshold).float()

            pred_mask = pred_mask.cpu().numpy()
            control_img = (control_img.numpy() + 1) / 2

            ## iou score
            # iou_score = mean_IoU(control_img[0, 0], pred_mask[0, 0], 2)
            # iou_score = IoU(pred_mask[0, 0], control_img[0, 0])
            # iou_scores.append(iou_score)

            data["control_img"] = Image.fromarray(
                (pred_mask[0, 0] * 255).astype(np.uint8)
            )

        # prepare text prompt
        inputs = processor(data["blip_img"], return_tensors="pt").to(
            "cuda", torch.float16
        )
        generate_id = blip2.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generate_id, skip_special_tokens=True)[
            0
        ].strip()

        if args.additional_prompt == None:
            generated_text += (
                ", dynamic lighting, intense light source, prominent lens flare, best quality, high resolution, masterpiece, intricate details"
                # ", full light sources with lens flare, best quality, high resolution"
            )
        else:
            generated_text += ", " + args.additional_prompt

        # Blur mask
        # data["mask_img"] = data["mask_img"].filter(ImageFilter.GaussianBlur(15))

        # denoise
        result = pipe(
            prompt=generated_text,
            # negative_prompt="NSFW, (word:1.5), watermark, blurry, missing body, amputation, mutilation",
            image=data["input_img_matching"],
            mask_image=data["mask_img"],
            control_image=(
                data["control_img"] if args.controlnet_path is not None else None
            ),
            num_inference_steps=args.step,
            guidance_scale=args.cfg_scale,
            generator=generator,
            repeat_time=args.noise_reinjection,
        ).images[0]

        # save result
        result = np.array(result)
        input_img = np.array(data["input_img"])
        box = data["box"]
        output_name = data["output_name"]

        input_img2 = result.copy()
        input_img2[box[2] : box[3] + 1, box[0] : box[1] + 1] = input_img[
            box[2] : box[3] + 1, box[0] : box[1] + 1
        ]

        result = blend_with_alpha(result, input_img2, box, blur_size=31)

        result = Image.fromarray(result.astype(np.uint8))
        result.save(f"{args.output_dir}/{output_name}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
