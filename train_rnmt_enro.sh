#!/bin/bash
# multi-gpu
#-m torch.distributed.launch --master_port 23333 --nproc_per_node 4 

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 2334 --nproc_per_node 2 train.py --ddp-backend=no_c10d /home/shanyong/wmt16_en_ro/fairseq_databin \
    --arch rnmt_base_nist --share-all-embeddings \
    --optimizer adam --adam-eps 1e-6 --clip-norm 5.0 \
    --lr-scheduler chen --replicas-num 2 --warmup-steps 500 --start-step 8000 --end-step 64000 \
    --lr 0.001 --min-lr 1e-9 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
	--max-tokens 4096 --save-dir checkpoints/rnmt_base_enro \
    --tensorboard-logdir events/rnmt_base_enro \
    --max-epoch 30 \
	--update-freq 1 --no-progress-bar --log-format json --log-interval 50 | tee logs/rnmt_base_enro.log
