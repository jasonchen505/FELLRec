for seed in 1
do
    for lr in 1e-4
    do
        for dropout in 0.05    
        do
            for sample in 1024
            do
                echo "lr: $lr, dropout: $dropout , seed: $seed,"
                CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate.yaml finetune.py \
                    --base_model " " \
                    --train_data_path "[\"./data/games/train_1024_user.json\"]"   \
                    --val_data_path "[\"./data/games/valid_5000_user.json\"]" \
                    --test_data_path "[\"./data/games/test_user.json\"]" \
                    --output_dir ./model/games/${seed}_${sample} \
                    --batch_size 64 \
                    --micro_batch_size 4 \
                    --num_epochs 75 \
                    --client_num 5 \
                    --round 5 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --alpha 0.7 \
                    --beta 1 \
                    --k 20 \
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint None \
                    --seed $seed \
                    --sample $sample \ 
            done    
        done
    done
done