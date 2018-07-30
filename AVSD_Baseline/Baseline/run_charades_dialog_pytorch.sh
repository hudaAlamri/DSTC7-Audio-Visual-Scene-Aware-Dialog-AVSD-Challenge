#!/bin/bash

#. path.sh
use_slurm=false
stage=1
slurm_queue=clusterNew
workdir=`pwd`
datadir=$workdir/data/charades
ftype="i3d"
#ftype="vggnet i3d c3d mfcc" # resnet"
in_size="2048" #feat dims: vggnet: 4096, i3d: 2048"

# model structure
modeltype=arsgmm
#modeltype=arsg
enc_psize="512 512 512 512 512" # use 512 for all features.
enc_hsize="0 0 0 0 0" # use 256 for mfcc. rest 0
dec_psize=256
dec_hsize=512
att_size=128

# audio feature params
winlen=0.04
winstep=0.02
winsize=30
winshift=20

# training params
batch_size=32 #20
max_length=100
#optimizer=Adam
optimizer=AdaDelta
L2_weight=0.0005
seed=1
num_epoch=1 #12

# generator params
beam=5
penalty=0.0

. utils/parse_options.sh || exit 1;

# directory and feature file setting
enc_psize_=`echo $enc_psize|sed "s/ /-/g"`
enc_hsize_=`echo $enc_hsize|sed "s/ /-/g"`
ftype_=`echo $ftype|sed "s/ /-/g"`

expdir=exp/charades_${modeltype}_${ftype_}_${optimizer}_ep${enc_psize_}_eh${enc_hsize_}_dp${dec_psize}_dh${dec_hsize}_att${att_size}_bs${batch_size}_L2w${L2_weight}_seed${seed}
train_opt=""

# command settings
if [ $use_slurm = true ]; then
  train_cmd="srun -p $slurm_queue --job-name train -X --chdir=$workdir --gres=gpu:1 "
  test_cmd="srun -p $slurm_queue --job-name test -X --chdir=$workdir --gres=gpu:1 "
  gpu_id=0
else
  train_cmd=""
  test_cmd=""
  gpu_id=2 #`utils/get_available_gpu_id.sh`
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
#set -x

# feature extraction
cnt=0
for feature in $ftype; do
    feafile=$datadir/charades2text_${feature}_features.pkl
    feafiles[cnt]=$feafile
    cnt=$((cnt + 1))
done


# training phase
mkdir -p $expdir
if [ $stage -le 2 ]; then
    echo start training
    $train_cmd local/train_video_dialog_model_pytorch.py \
      --gpu $gpu_id \
      --optimizer $optimizer \
      --L2-weight $L2_weight \
      --type $modeltype \
      --train $datadir/train.json \
      --valid $datadir/valid.json \
      --test $datadir/test.json \
      --feafile ${feafiles[@]} \
      --capfile $datadir/CAP.json \
      --vocabfile $datadir/worddict.json \
      --batch-size $batch_size \
      --max-length $max_length \
      --learn-decay 1.0 \
      --model $expdir/caption_model \
      --in-size $in_size \
      --enc-psize $enc_psize \
      --enc-hsize $enc_hsize \
      --dec-psize $dec_psize \
      --dec-hsize $dec_hsize \
      --att-size $att_size \
      --rand-seed $seed \
      --num-epoch $num_epoch \
      $train_opt \
      |& tee $expdir/train.log

fi

echo 
# testing phase
if [ $stage -le 3 ]; then
    gpu_id=0
    for target in valid test; do
        echo start caption generation for $target set
	result=${expdir}/generate_${target}_b${beam}_p${penalty}.result
        $test_cmd local/generate_video_dialog_pytorch.py \
          --gpu $gpu_id \
          --test $datadir/${target}.json \
          --feafile ${feafiles[@]} \
          --capfile $datadir/CAP.json \
          --model $expdir/caption_model_best \
          --beam $beam \
          --penalty $penalty \
         |& tee $result 
    done
fi

# scoring
if [ $stage -le 4 ]; then
    for target in valid test; do
        echo start evaluation for $target set
	result=${expdir}/generate_${target}_b${beam}_p${penalty}.result
        reference=./data/charades/charades_${target}_ref.json
        if [ ! -f $reference ]; then
            utils/get_annotation_charades.py \
                  $datadir/CAP.json $datadir/${target}.json $reference
        fi
        result_json=${result%.*}.json
        result_eval=${result%.*}.eval
        echo Evaluating: $result
        utils/get_hypotheses.py $result $result_json
        utils/evaluate.py $reference $result_json >& $result_eval
        echo Wrote details in $result_eval
        awk '{if (NR>=26 && NR<33) {print $0}}' $result_eval
    done
fi

