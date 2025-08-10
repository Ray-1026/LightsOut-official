CUDA_VISIBLE_DEVICES=1 python inference.py \
    --sd_path pretrained/sd_outpainting \
    --light_regress_path pretrained/light_regress/model_61_0.63.pth \
    --controlnet_path pretrained/controlnet \
    --lora_path pretrained/lora_sd \
    --blip2_path pretrained/blip2/blip2 \
    --blip2_proc_path pretrained/blip2/blip_processor \
    --output_dir res \
    --dataset_config configs/flare7kpp_dataset.yml \
    --seed 42
