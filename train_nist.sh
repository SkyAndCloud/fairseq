#!/bin/bash
# multi-gpu
#-m torch.distributed.launch --master_port 23333 --nproc_per_node 4 

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 23333 --nproc_per_node 2 train.py --ddp-backend=no_c10d ../kernel_attention/data-bin/NIST_zh_en_1.25m \
	  --arch rnmt_base_nist \
	    --optimizer adam --clip-norm 5.0 \
	      --lr-scheduler chen --replicas-num 2 --warmup-steps 500 --start-step 8000 --end-step 64000 \
	        --lr 0.001 --min-lr 1e-09 \
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
		    --max-tokens 4096 --save-dir checkpoints/rnmt_base_nist \
            --tensorboard-logdir events/rnmt_base_nist \
            --max-epoch 25 \
		    --update-freq 1 --no-progress-bar --log-format json --log-interval 50 | tee logs/rnmt_base_nist.log
