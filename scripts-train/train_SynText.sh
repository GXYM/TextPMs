#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_TextPMs.py --exp_name Synthtext --net resnet50 --lr 0.001 --input_size 512 --batch_size 10 --gpu 0 --max_epoch 1 --save_freq 1 --viz --num_workers 24
