# !/bin/bash
CKPT_DIR=$1
OUT_DIR=$2
GPU=$3

mkdir -p $2

declare -a testset=('valid' 'test')
declare -a refset=(
                '/home4/shanyong/wmt16/newstest2013.tok.atat.de'
                '/home4/shanyong/wmt16/newstest2014.tok.atat.de'
                )
len=${#testset[@]}
max=`echo $len-1|bc`
MOSES=/home4/shanyong/mosesdecoder
BEAM=10
ALPHA=0
BATCH=100
DATABIN=../kernel_attention/data-bin/wmt16_en_de_bpe32k_pretrain

for m in $CKPT_DIR/*.pt; do
    if [ ! -f $m ]; then
        continue
    fi
    m_name=`basename ${m}`
    echo $m_name
    sum=0
    for i in $(seq 0 $max); do
        t=${testset[$i]}
        r=${refset[$i]}
        CUDA_VISIBLE_DEVICES=$GPU python generate.py $DATABIN --path $m --gen-subset $t --beam $BEAM --batch-size $BATCH --remove-bpe --lenpen $ALPHA --log-format=none > $OUT_DIR/${m_name}.$t
        grep ^H $OUT_DIR/${m_name}.$t | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > $OUT_DIR/${m_name}.$t.out
        perl $MOSES/scripts/tokenizer/replace-unicode-punctuation.perl -l de < $OUT_DIR/${m_name}.$t.out > $OUT_DIR/${m_name}.$t.out.n
        perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $OUT_DIR/${m_name}.$t.out.n > $OUT_DIR/${m_name}.$t.out.n.atat
        bleu=`perl ${MOSES}/scripts/generic/multi-bleu.perl $r < $OUT_DIR/${m_name}.$t.out.n.atat`
        echo -e "\t$t\t$bleu"
    done    
done
