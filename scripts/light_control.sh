export CUDA_VISIBLE_DEVICES=2

# export MODEL_DIR="stabilityai/stable-diffusion-2-inpainting"
export MODEL_DIR="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR="weights/light_control"

accelerate launch train_light_control.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
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
  # --num_train_epochs=5 \

# accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
#  --output_dir="model_out" \
#  --dataset_name=multimodalart/facesyntheticsspigacaptioned \
#  --conditioning_image_column=spiga_seg \
#  --image_column=image \
#  --caption_column=image_caption \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --validation_image "./face_landmarks1.jpeg" "./face_landmarks2.jpeg" "./face_landmarks3.jpeg" \
#  --validation_prompt "High-quality close-up dslr photo of man wearing a hat with trees in the background" "Girl smiling, professional dslr photograph, dark background, studio lights, high quality" "Portrait of a clown face, oil on canvas, bittersweet expression" \
#  --train_batch_size=4 \
#  --num_train_epochs=3 \
#  --tracker_project_name="controlnet" \
#  --enable_xformers_memory_efficient_attention \
#  --checkpointing_steps=5000 \
#  --validation_steps=5000 \