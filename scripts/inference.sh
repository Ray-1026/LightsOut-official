export CUDA_VISIBLE_DEVICES=1

export MODEL_DIR="stabilityai/stable-diffusion-2-inpainting"
# export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-inpainting"
export OUTPUT_DIR="res"

python inference.py \
    --sd_path $MODEL_DIR \
    --blip2_path Salesforce/blip2-opt-2.7b \
    --light_outpaint_path weights/light_outpaint \
    --light_control_path weights/light_control \
    --light_regress_path pretrained/light_regress/model.pth \
    --dataset_config configs/flare7kpp_dataset.yml \
    --output_dir $OUTPUT_DIR \
    --cfg_scale 5.0 \
    --seed 42
