# !/bin/bash
CKPT_DIR=$1
OUT_DIR=$2
GPU=$3

mkdir -p $2

declare -a testset=('valid' 'test')
declare -a refset=(
                '/home/shanyong/wmt16_en_ro/dev/other/dev.bpe.r2l.debpe.ro'
                '/home/shanyong/wmt16_en_ro/test/other/test.bpe.r2l.debpe.ro'
                )
len=${#testset[@]}
max=`echo $len-1|bc`
script='/home2/yshan/moses-scripts/scripts/generic/multi-bleu.perl'
BEAM=10
ALPHA=0.0
BATCH=250
DATABIN=/home/shanyong/wmt16_en_ro/fairseq_databin

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
        bleu=`perl $script -lc $r < $OUT_DIR/${m_name}.$t.out`
        num=`echo $bleu|sed "s/.*BLEU\ =\ \([0-9.]\{1,\}\).*/\1/"`
        sum=`echo "scale=2;$sum+$num"|bc`
        echo -e "\t$t\t$bleu"
    done    
    avg=`echo "scale=2;$sum/$len"|bc`
    echo -e "\tVTAVG\t$avg"
done
