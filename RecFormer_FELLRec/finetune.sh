CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate.yaml finetune.py \
    --pretrain_ckpt pretrain_ckpt/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/finetune_data/games \
    --num_train_epochs 128 \
    --batch_size 8 \
    --alpha 0.7 \
    --beta 5 \
    --k 10 \
    --fp16 \
    --finetune_negative_sample_size -1