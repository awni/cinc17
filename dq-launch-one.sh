#!/bin/bash

echo This job is running on $(hostname)
export LD_LIBARY_PATH=/usr/local/cuda-8.0/lib64
export PATH=/usr/local/cuda-8.0/bin
python train.py -c cnn2-size256_128-num64_64.json
echo "Started training at: ", $(date +%m:%d:%y-%H:%M:%S)
