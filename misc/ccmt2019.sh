#!/usr/bin/env bash

python -u preprocess.py -s zh -t en --srcdict /home2/wshugen/ccmt2019/trainsformer_big/dict.zh.txt --tgtdict /home2/wshugen/ccmt2019/trainsformer_big/dict.en.txt --trainpref /home2/wshugen/ccmt2019/end2end/data/train/train --validpref /home2/wshugen/ccmt2019/end2end/data/valid/valid --destdir /home2/wshugen/ccmt2019/end2end/data-bin --workers 1


python -u train.py /home2/wshugen/ccmt2019/end2end/data-bin --audio-path /home2/wshugen/ccmt2019/end2end/dump --audio-encoder /home2/wshugen/ccmt2019/var/checkpoints/common_voice_encoder_mtalpha0.5_lm0.1.pt --arch transformer_wmt_en_de_big_t2t --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 512 --update-freq 1 --no-progress-bar --log-format json --log-interval 10 --save-interval-updates 1000 --keep-interval-updates 100 --save-dir /home2/wshugen/ccmt2019/var/checkpoints/t2t_big --tensorboard-logdir /home2/wshugen/ccmt2019/var/t2t_big --reset-optimizer --reset-lr-scheduler
