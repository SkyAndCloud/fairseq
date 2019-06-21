#!/bin/bash

if [[ -z ${YSHAN_EVAL_PATH} ]]; then
    YSHAN_EVAL_PATH=/home/shanyong/newhdd_sl/ccmt2019_speech/eval_bleu_sbp.sh
fi
if [[ ! -f ${YSHAN_EVAL_PATH} ]]; then
    echo "YSHAN_EVAL_PATH does NOT exist"
    exit
fi

if [[ -z ${YSHAN_SRC_SGM} ]]; then
    YSHAN_SRC_SGM=/home/newhdd/yshan/ccmt2019_speech/data/zh_en/preprocessed/valid/asr.no_\#.valid.src.sgm
fi
if [[ ! -f ${YSHAN_SRC_SGM} ]]; then
    echo "YSHAN_SRC_SGM does NOT exist"
    exit
fi

if [[ -z ${YSHAN_REF_SGM} ]]; then
    YSHAN_REF_SGM=/home/newhdd/yshan/ccmt2019_speech/data/zh_en/preprocessed/valid/asr.no_\#.valid.ref.sgm
fi
if [[ ! -f ${YSHAN_REF_SGM} ]]; then
    echo "YSHAN_REF_SGM does NOT exist"
    exit
fi

if [[ -z ${DATA_BIN} ]]; then
  DATA_BIN=/home2/wshugen/ccmt2019/end2end/data-bin
fi
echo "Data bin: [${DATA_BIN}]"

if [[ -z ${GEN_SUBSET} ]]; then
    GEN_SUBSET=valid
fi
echo "Subsets: [${GEN_SUBSET}]"

if [[ -z ${BEAM_SIZE} ]]; then
    BEAM_SIZE=5
fi

if [[ -z ${LEN_PEN} ]]; then
    LEN_PEN=1.0
fi

if [[ -z ${MODEL_PATHS} ]]; then
    MODEL_PATHS=/home2/wshugen/git_fairseq/ensemble/mt1_checkpoint31.pt:/home2/wshugen/git_fairseq/ensemble/mt2_checkpoint26.pt
fi
echo "Model paths: [${MODEL_PATHS}]"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <WEIGHTS> [METHOD=sum] [DEVICE_ID=0]"
  exit
fi

WEIGHTS="$1"
METHOD=sum
if [[ $# -ge 2 ]]; then
  METHOD=$2
fi

DEVICE_ID=0
if [[ $# -ge 3 ]]; then
  DEVICE_ID=$3
fi
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

_OUTPUT_DIR=./out/${METHOD}_${WEIGHTS}_${RANDOM}
_OUTPUT_DIR=${_OUTPUT_DIR/\'\(/} # sub '( with blank
_OUTPUT_DIR=${_OUTPUT_DIR/\)\'/} # sub )' with blank
_OUTPUT_DIR=${_OUTPUT_DIR//,/_}  # sub , with _
OUTPUT_DIR=${_OUTPUT_DIR}
if [[ ! -d ${OUTPUT_DIR} ]]; then
  mkdir -p ${OUTPUT_DIR}
fi
echo "Ensemble with method [${METHOD}] weights: [${WEIGHTS}] on [GPU${DEVICE_ID}] to ${OUTPUT_DIR} for [${MODEL_PATHS}], beam ${BEAM_SIZE}, len penalty ${LEN_PEN}" | tee ${OUTPUT_DIR}/meta

OVERRIDE_OPTION=''
if [[ ! -z ${MODEL_OVERRIDES} ]]; then
    OVERRIDE_OPTION='--model-overrides ${MODEL_OVERRIDES}'
fi

RESULT_PATH=${OUTPUT_DIR}/res
cmd="python generate.py ${DATA_BIN} --path ${MODEL_PATHS} ${OVERRIDE_OPTION} --beam ${BEAM_SIZE} --batch-size 8 --gen-subset ${GEN_SUBSET} --remove-bpe  --lenpen ${LEN_PEN} --ensemble-weights ${WEIGHTS} --ensemble-method ${METHOD} | tee ${RESULT_PATH}"
echo ${cmd}
eval ${cmd}

TARGET_PATH=${RESULT_PATH}.out
cmd="grep ^H ${RESULT_PATH} | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > ${TARGET_PATH}"
echo ${cmd}
eval ${cmd}

cmd="bash ${YSHAN_EVAL_PATH} ${TARGET_PATH} ${YSHAN_SRC_SGM} ${YSHAN_REF_SGM} >${TARGET_PATH}.rs"
echo ${cmd}
eval ${cmd}
