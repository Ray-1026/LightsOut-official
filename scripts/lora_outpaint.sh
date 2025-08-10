export CUDA_VISIBLE_DEVICES=1

# export MODEL_DIR="stabilityai/stable-diffusion-2-inpainting"
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-inpainting"

accelerate launch --num_processes 1 train_light_outpaint.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --dataset_name="" \
  --instance_prompt="nighttime, with full light source and lens flare artifacts" \
  --mixed_precision="fp16" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --seed=0 \
  --validation_epochs=1 \
  --output_dir="weights/lora_sd1.5" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --dataloader_num_workers=4 \
  --max_train_steps=25000
