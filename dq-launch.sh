#!/bin/bash

echo This job is running on $(hostname)
export LD_LIBARY_PATH=/usr/local/cuda-8.0/lib64
export PATH=/usr/local/cuda-8.0/bin
python train.py -c $DQ_CFG
echo "Started training at: ", $(date +%m:%d:%y-%H:%M:%S)
