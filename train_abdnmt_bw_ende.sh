#!/bin/bash
# multi-gpu
#-m torch.distributed.launch --master_port 23333 --nproc_per_node 4 

python train.py \
    data-bin/wmt16_en_de_bpe32k_pretrain \
    --arch abdnmt_base_ende --bd-write --agree-weight 0 \
    --pretrain-file ../abdnmt_only_bd_ende/checkpoint28.pt \
    --optimizer adam --adam-eps 1e-6 --clip-norm 5.0 \
    --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --lr 0.001 --min-lr 5e-5 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 2884 --save-dir checkpoints/abdnmt_bw_ende \
    --save-interval-updates 1500 --keep-interval-updates 50 \
    --tensorboard-logdir events/abdnmt_bw_ende \
    --max-epoch 30 --lazy-load --num-workers 4 --fp16 --ddp-backend=no_c10d \
    --update-freq 11 --no-progress-bar --log-format json --log-interval 1 | tee -a logs/abdnmt_bw_ende.log
