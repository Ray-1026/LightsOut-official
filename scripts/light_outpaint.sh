export CUDA_VISIBLE_DEVICES=1

export MODEL_DIR="stabilityai/stable-diffusion-2-inpainting"
# export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-inpainting"
export OUTPUT_DIR="weights/light_outpaint_lora"

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
