#!/bin/bash
# multi-gpu
#-m torch.distributed.launch --master_port 23333 --nproc_per_node 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 23334 --nproc_per_node 4 train.py --ddp-backend=no_c10d ../kernel_attention/data-bin/wmt16_en_de_bpe32k_pretrain \
	  --arch rnmt_base_ende \
	    --optimizer adam --adam-eps 1e-6 --clip-norm 5.0 \
	      --lr-scheduler chen --replicas-num 8 --warmup-steps 50 --start-step 200000 --end-step 1200000 \
	        --lr 0.0001 --min-lr 1e-9 \
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
		    --max-tokens 4096 --save-dir checkpoints/rnmt_base_ende \
            --tensorboard-logdir events/rnmt_base_ende \
            --max-epoch 30 \
		    --update-freq 2 --no-progress-bar --log-format json --log-interval 50 | tee logs/rnmt_base_ende.log
