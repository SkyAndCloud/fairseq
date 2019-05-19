#!/bin/bash
# multi-gpu
#-m torch.distributed.launch --master_port 23333 --nproc_per_node 4 --ddp-backend=no_c10d

#--pretrain-file ../abdnmt_only_bd_enro/checkpoint_best.pt \
model=abdnmt_bw_l2_enro
#CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 23334 --nproc_per_node 2 train.py /home/shanyong/wmt16_en_ro/fairseq_databin \
CUDA_VISIBLE_DEVICES=0 python train.py /home/shanyong/wmt16_en_ro/fairseq_databin \
    --ddp-backend no_c10d \
    --arch abdnmt_base_nist --bd-write --share-all-embeddings \
    --restore-file ../abdnmt_bw_enro/checkpoint_best.pt --reset-optimizer --reset-lr-scheduler \
    --save-interval-updates 1000 --keep-interval-updates 45 \
	--optimizer adam --adam-eps 1e-6 --clip-norm 5.0 \
	--lr-scheduler chen --replicas-num 2 --warmup-steps 500 --start-step 8000 --end-step 64000 \
    --lr 0.001 --min-lr 1e-9 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
    --max-tokens 2048 --save-dir checkpoints/$model \
    --tensorboard-logdir events/$model \
    --max-epoch 30 \
    --update-freq 2 --no-progress-bar --log-format json --log-interval 100 | tee logs/$model.log
