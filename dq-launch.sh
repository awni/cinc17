#!/bin/bash

#gpu=3
#env CUDA_VISIBLE_DEVICES=$gpu 
python train.py -c $DQ_CFG
#configs/cnn20-lr-1.json -v
echo "Started training at: ", $(date +%m:%d:%y-%H:%M:%S)
