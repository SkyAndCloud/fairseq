#!/bin/bash
# multi-gpu
#-m torch.distributed.launch --master_port 23333 --nproc_per_node 4 

python -m torch.distributed.launch --master_port 23333 --nproc_per_node 4 train.py \
    ../kernel_attention/data-bin/wmt16_en_de_bpe32k_pretrain \
    --arch abdnmt_base_ende --agree-weight 0 \
    --pretrain-file ../abdnmt_only_bd_ende/checkpoint28.pt \
    --optimizer adam --adam-eps 1e-6 --clip-norm 5.0 \
    --lr-scheduler chen --replicas-num 8 --warmup-steps 50 --start-step 200000 --end-step 1200000 \
    --lr 0.0001 --min-lr 1e-9 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --save-dir checkpoints/abdnmt_ende \
    --save-interval-updates 1000 --keep-interval-updates 50 \
    --tensorboard-logdir events/abdnmt_ende \
    --max-epoch 30 --lazy-load --num-workers 4 --ddp-backend=no_c10d \
    --update-freq 2 --no-progress-bar --log-format json --log-interval 1 | tee -a logs/abdnmt_ende.log
