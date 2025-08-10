export CUDA_VISIBLE_DEVICES=3

python train_light_regress.py \
        --model light_regress \
        --config configs/flare7kpp_dataset.yml \
        --batch_size 32 \
        --epochs 60 \
        --lr 0.0001 \
        --num_workers 32 \
        --save_ckpt_dir weights/light_regress

python train_light_regress.py \
        --model unet_3maps \
        --config configs/flare7kpp_dataset.yml \
        --batch_size 8 \
        --epochs 25 \
        --lr 0.0001 \
        --num_workers 8 \
        --save_ckpt_dir weights/light_3maps
