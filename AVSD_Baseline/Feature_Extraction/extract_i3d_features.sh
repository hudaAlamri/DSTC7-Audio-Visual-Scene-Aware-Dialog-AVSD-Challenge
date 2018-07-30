#!/bin/bash

use_slurm=true
slurm_queue=clusterNew
workdir=`pwd`
frame_stride=4
seqence_length=16
featype="Mixed_5c"
modeldir=$workdir/i3d_model
output_folder=$workdir/i3d_features_stride${frame_stride}
modelpath=$modeldir/data/checkpoints/rgb_imagenet
imgpath=Charades_v1_rgb

# download a i3d model
if [ ! -d $modeldir ]; then
    mkdir -p $modeldir
    git clone https://github.com/deepmind/kinetics-i3d $modeldir
    cp $modeldir/i3d.py i3d.py
fi

# command settings
if [ $use_slurm == true ]; then
  cmd="srun -X --gres=gpu:1 -p $slurm_queue python"
  gpu_id=0
else
  cmd=""
  gpu_id=`utils/get_available_gpu_id.sh`
fi

if [ ! -d $output_folder ]; then
    mkdir -p $output_folder
    # extract features
    $cmd extract_i3d_rgb_features.py \
          --input $imgpath \
          --net_output $featype \
          --feature_dim 2048 \
          --model_path $modelpath\
          --stride $frame_stride  --output $output_folder --seq_length $seqence_length
fi



