#!/bin/bash
# multi-gpu
#-m torch.distributed.launch --master_port 23333 --nproc_per_node 4 --ddp-backend=no_c10d

#--pretrain-file ../abdnmt_only_bd_nist/checkpoint24.pt \
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --master_port 23333 --nproc_per_node 3 train.py ../kernel_attention/data-bin/NIST_zh_en_1.25m \
    --ddp-backend no_c10d \
    --arch abdnmt_base_nist --bd-write \
    --save-interval-updates 1500 --keep-interval-updates 45 \
	--optimizer adam --adam-eps 1e-6 --clip-norm 5.0 \
	--lr-scheduler chen --replicas-num 2 --warmup-steps 500 --start-step 8000 --end-step 64000 \
    --lr 0.001 --min-lr 1e-9 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 1792 --save-dir checkpoints/abdnmt_bw_nist \
    --tensorboard-logdir events/abdnmt_bw_nist \
    --max-epoch 30 \
    --update-freq 2 --no-progress-bar --log-format json --log-interval 1 | tee -a logs/abdnmt_bw_nist.log
