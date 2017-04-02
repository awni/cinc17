#!/bin/bash

#gpu=3
#env CUDA_VISIBLE_DEVICES=$gpu 
python train.py -c configs/explore.json 
echo "Started training at: ", $(date +%m:%d:%y-%H:%M:%S)
