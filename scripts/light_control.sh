export CUDA_VISIBLE_DEVICES=0

export MODEL_DIR="stabilityai/stable-diffusion-2-1-base"
# export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="weights/light_control"

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
