import argparse
import numpy as np
import cv2
import os
import torch
import torchvision
from PIL import Image, ImageFilter
from tqdm import tqdm
from diffusers import ControlNetModel, DPMSolverMultistepScheduler
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from src.pipelines.pipeline_stable_diffusion_outpaint import OutpaintPipeline
from src.pipelines.pipeline_controlnet_outpaint import ControlNetOutpaintPipeline
from src.schedulers.scheduling_pndm import CustomScheduler
from src.models.unet import U_Net
from src.models.light_source_regressor import LightSourceRegressor
from utils.dataset import CustomImageLoader
from utils.utils import (
    blend_with_alpha,
    load_mfdnet_checkpoint,
    predict_flare_from_6_channel,
    predict_flare_from_3_channel,
    blend_light_source,
)
from SIFR_models.flare7kpp.model import Uformer
from SIFR_models.mfdnet.model import Model


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--sd_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--blip2_path",
        type=str,
        default="Salesforce/blip2-opt-2.7b",
        help="Path to pretrained blip2 model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--light_regress_path",
        type=str,
        default="./pretrained/light_regress/model.pth",
        help="Path to pretrained light regress model.",
    )
    parser.add_argument(
        "--light_control_path",
        type=str,
        default="./pretrained/light_control",
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--light_outpaint_path",
        type=str,
        default="./pretrained/light_outpaint_lora",
        help="Path to lora model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="res",
        help="The output directory where predictions are saved",
    )
    # parser.add_argument(
    #     "--dataset_config",
    #     type=str,
    #     default="configs/flare7kpp_dataset.yml",
    #     help="Path to dataset configuration file.",
    # )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible inference."
    )
    # parser.add_argument(
    #     "--light_setting",
    #     choices=["incomplete", "no"],
    #     default="no",
    #     help="Light setting for the model.",
    # )
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

    # add for custom data inference
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default="custom_data",
        help="Path to custom data for inference.",
    )
    parser.add_argument(
        "--left_outpaint",
        type=int,
        default=64,
        help="Left outpaint size for custom data (suggest no more than 128).",
    )
    parser.add_argument(
        "--right_outpaint",
        type=int,
        default=64,
        help="Right outpaint size for custom data (suggest no more than 128).",
    )
    parser.add_argument(
        "--up_outpaint",
        type=int,
        default=64,
        help="Up outpaint size for custom data (suggest no more than 128).",
    )
    parser.add_argument(
        "--down_outpaint",
        type=int,
        default=64,
        help="Down outpaint size for custom data (suggest no more than 128).",
    )

    # flare removal
    parser.add_argument(
        "--remove_flare", action="store_true", help="Remove flare from images."
    )
    parser.add_argument(
        "--SIFR_model",
        type=str,
        choices=["flare7k++", "mfdnet"],
        default=None,
        help="Select the SIFR model to use.",
    )
    parser.add_argument(
        "--SIFR_model_path", type=str, default=None, help="Path to the SIFR model."
    )

    args = parser.parse_args()

    if args.remove_flare:
        if args.SIFR_model is None or args.SIFR_model_path is None:
            raise ValueError(
                "SIFR model and model path must be specified when removing flare."
            )

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
    processor = Blip2Processor.from_pretrained(
        blip2_path, revision="51572668da0eb669e01a189dc22abe6088589a24"
    )
    blip2 = Blip2ForConditionalGeneration.from_pretrained(
        blip2_path,
        torch_dtype=torch_dtype,
        revision="51572668da0eb669e01a189dc22abe6088589a24",
    )
    blip2 = blip2.to(device)
    return processor, blip2


def build_SIFR_model(SIFR_model, SIFR_model_path, device="cuda"):
    if SIFR_model == "flare7k++":
        model = Uformer(img_size=512, img_ch=3, output_ch=6).to(device)
        model.load_state_dict(torch.load(SIFR_model_path))
    elif SIFR_model == "mfdnet":
        model = Model().to(device)
        load_mfdnet_checkpoint(model, SIFR_model_path)

    model.eval()
    return model


def collate_fn(batch):
    return batch[0]


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/outpainted", exist_ok=True)

    # lightsource regress model
    lsr_module = LSRMInit(args.light_regress_path)

    # sd outpaint
    pipe = OutpainterInit(
        args.sd_path, args.light_control_path, args.light_outpaint_path
    )

    # blip2
    processor, blip2 = Blip2Init(args.blip2_path)

    # custom dataset
    dataset = CustomImageLoader(
        args.custom_data_path,
        left_outpaint=args.left_outpaint,
        right_outpaint=args.right_outpaint,
        up_outpaint=args.up_outpaint,
        down_outpaint=args.down_outpaint,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
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
    result_paths = []
    result_boxes = []
    # iou_scores = []

    for data in tqdm(dataset):
        # for data in tqdm(dataloader):
        with torch.no_grad():
            input_img = data["input_img"]

            input_img = transform(input_img).unsqueeze(0).to("cuda")

            pred_mask = lsr_module.forward_render(input_img)

            pred_mask = (pred_mask > threshold).float()

            pred_mask = pred_mask.cpu().numpy()

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
            image=data["input_img"],
            mask_image=data["mask_img"],
            control_image=(
                data["control_img"] if args.light_control_path is not None else None
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
        result.save(f"{args.output_dir}/outpainted/{output_name}")

        result_paths.append(f"{args.output_dir}/outpainted/{output_name}")
        result_boxes.append(box)

    if args.remove_flare:
        deflare_path = f"{args.output_dir}/deflare_res"
        crop_path = f"{args.output_dir}/deflare_crop"
        os.makedirs(deflare_path, exist_ok=True)
        os.makedirs(crop_path, exist_ok=True)

        sifr_model = build_SIFR_model(args.SIFR_model, args.SIFR_model_path)

        sifr_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
            ]
        )

        for p, box in tqdm(zip(result_paths, result_boxes)):
            img = Image.open(p).convert("RGB")
            img = sifr_transform(img).unsqueeze(0).cuda()

            with torch.no_grad():
                output_img = sifr_model(img)

                gamma = torch.Tensor([2.2])

                if args.SIFR_model == "flare7k++":
                    deflare_img, _, _ = predict_flare_from_6_channel(output_img, gamma)
                    torchvision.utils.save_image(
                        deflare_img, f"{deflare_path}/{os.path.basename(p)}"
                    )
                elif args.SIFR_model == "mfdnet":
                    flare_mask = torch.zeros_like(img)
                    deflare_img, _ = predict_flare_from_3_channel(
                        output_img, flare_mask, output_img, img, img, gamma
                    )
                    deflare_img = blend_light_source(img, deflare_img, 0.999)
                    torchvision.utils.save_image(
                        deflare_img, f"{deflare_path}/{os.path.basename(p)}"
                    )

            crop_img = deflare_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
            crop_img = np.clip(crop_img, 0.0, 1.0)
            crop_img = (crop_img * 255).astype(np.uint8)
            crop_img = crop_img[box[2] : box[3] + 1, box[0] : box[1] + 1, :]
            Image.fromarray(crop_img).resize((512, 512), Image.LANCZOS).save(
                f"{crop_path}/{os.path.basename(p)}"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
