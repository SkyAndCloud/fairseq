#!/bin/bash
# multi-gpu
#-m torch.distributed.launch --master_port 23333 --nproc_per_node 4 --ddp-backend=no_c10d

python -m torch.distributed.launch --master_port 23333 --nproc_per_node 4 train.py ../kernel_attention/data-bin/NIST_zh_en_1.25m \
    --ddp-backend=no_c10d \
    --arch abdnmt_base_nist --agree-weight 0 \
    --pretrain-file ../abdnmt_only_bd_nist_enc_no_drop/checkpoint_best.pt \
	--optimizer adam --adam-eps 1e-6 --clip-norm 5.0 \
	--lr-scheduler chen --replicas-num 2 --warmup-steps 500 --start-step 8000 --end-step 64000 \
    --lr 0.001 --min-lr 1e-9 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 2048 --save-dir checkpoints/abdnmt_nist \
    --tensorboard-logdir events/abdnmt_nist \
    --max-epoch 30 \
    --update-freq 1 --no-progress-bar --log-format json --log-interval 50 | tee -a logs/abdnmt_nist.log
