#!/usr/bin/env bash

work_path=$(dirname $0)
cd ./${work_path}

python ./eval_st40.py --network resnet50_v1d --dataset st40 --gpus 0 \
 --pretrained horelation_resnet50_v1d_st40_best.params --save-outputs