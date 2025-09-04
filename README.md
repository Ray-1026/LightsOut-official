<img src="https://ray-1026.github.io/lightsout/static/images/favicon_new.svg" height="100px" align="right">

# LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal

[Shr-Ruei Tsai](https://ray-1026.github.io/), [Wei-Cheng Chang](), [Jie-Ying Lee](https://jayinnn.dev/), [Chih-Hai Su](https://su-terry.github.io/), [Yu-Lun Liu](https://yulunalexliu.github.io/)

National Yang Ming Chaio Tung University

<!-- [![arXiv](https://img.shields.io/badge/)](https://arxiv.org/) -->

![Teaser](./assets/teaser.png)

## Environment Setup
```
conda create -n lightsout python=3.9
conda activate lightsout
pip install -r requirements.txt
```

## Dataset Preparation
1. Download the Flare7K++ dataset from [Flare7K](https://github.com/ykdai/Flare7K).
2. Unzip the dataset and place it in the `data/` directory.
3. Modify the `configs/flare7kpp_dataset.yml` file to set the correct paths for the dataset.

## Inference
Our inference script also provides the option to remove flare. You can go to the repositories [Flare7K](https://github.com/ykdai/Flare7K) or [MFDNet](https://github.com/Jiang-maomao/flare-removal) to download the pretrained models. Then, you can add the following arguments to the inference script:

```bash
--remove_flare \
--SIFR_model flare7k++ \
--SIFR_model_path /path/to/the/SIFR/model/pretrained/weights
```

Alternatively, you can directly use the provided script in the `scripts/` folder. Just modify the options within the script as needed:
```bash
bash scripts/inference.sh
```

## Evaluation
After running inference on the Flare7k++ test set, a folder will be generated containing the outpainted images. The structure of the folder will look like this:

```
/path/to/your/output/dir
|── deflare_crop/ # Cropped images with flare removed
|   ├── xxx.png
|   └── ...
|
|── deflare_res/ # Full images with flare removed (for evaluation)
|   ├── xxx.png
|   └── ...
|
|── xxx.png
|── ...
```

Evaluate your results using the provided evaluation script:

```bash
python evaluate.py \
    --input /path/to/your/flare/removal/results/in/`deflare_res` \
    --gt /path/to/flare7kpp/test/gt \
    --mask /path/to/flare7kpp/test/mask \
    --crop_margin 0 # 0 => no light source, 15 => incomplete light source
```

## Inference on Custom Data
To run inference on your own images, you can use the `inference_custom.py` script.

```bash
python inference_custom.py \
    --light_outpaint_path /path/to/your/light/outpaint/weights \
    --light_control_path /path/to/your/light/control/weights \
    --light_regress_path /path/to/your/light/regress/weights \
    --custom_data_path /path/to/your/custom/data \
    --output_dir /path/to/save/outpainted/images \  # Change this to your desired path
    --left_outpaint 64 \
    --right_outpaint 64 \
    --up_outpaint 64 \
    --down_outpaint 64 \
    --remove_flare \  # add this flag to remove flare, if you don't want to remove flare at the same time, just omit this line
    --SIFR_model flare7k++ \
    --SIFR_model_path /path/to/the/SIFR/model/pretrained/weights
```

## Training
All components of the pipeline are trained separately:

1. **Train the Light Source Regression Module**
  ```bash
  python train_light_regress.py \
        --model light_regress \
        --config configs/flare7kpp_dataset.yml \
        --batch_size 32 \
        --epochs 100 \
        --lr 0.0001 \
        --num_workers 32 \  # Adjust based on your system
        --save_ckpt_dir /path/to/save/light_regress # Change this to your desired path
  ```
  
  Alternatively, you can use the provided script:

  ```bash
  bash scripts/light_regress.sh
  ```

2. **Train the Light Source Condition Module**
  ```bash
  export MODEL_DIR="stabilityai/stable-diffusion-2-1-base"
  export OUTPUT_DIR="/path/to/save/light_control"  # Change this to your desired path

  accelerate launch train_light_control.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --blip2_model_path="Salesforce/blip2-opt-2.7b" \
    --output_dir=$OUTPUT_DIR \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2500 \
    --validation_steps=1000000 \
    --mixed_precision="fp16" \
    --instance_prompt=", full light sources with lens flare, best quality, high resolution" \
    --dataloader_num_workers=8 \
    --gradient_checkpointing \
    --max_train_steps=20000
  ```

  Alternatively, you can use the provided script:

  ```bash
  bash scripts/light_control.sh
  ```

3. **Train the Light Source Outpainter**
  ```bash
  export MODEL_DIR="stabilityai/stable-diffusion-2-inpainting"
  export OUTPUT_DIR="/path/to/save/light_outpaint_lora"  # Change this to your desired path

  accelerate launch --num_processes 1 train_light_outpaint.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --blip2_model_path="Salesforce/blip2-opt-2.7b" \
    --dataset_config_name="configs/flare7kpp_dataset.yml" \
    --instance_prompt=", full light sources with lens flare, best quality, high resolution" \
    --mixed_precision="fp16" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --checkpointing_steps=2500 \
    --learning_rate=1e-05 \
    --lr_scheduler="constant" \
    --seed=0 \
    --output_dir=$OUTPUT_DIR \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --dataloader_num_workers=8 \
    --max_train_steps=25000
  ```

  Alternatively, you can use the provided script:

  ```bash
  bash scripts/light_outpaint.sh
  ```

## Citation
```
@InProceedings{tsai2025lightsout,
  title     = {LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal},
  author    = {Tsai, Shr-Ruei and Chang, Wei-Cheng and Lee, Jie-Ying and Su, Chih-Hai and Liu, Yu-Lun},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2025},
}
```

## Acknowledgements
This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-E-A49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.
