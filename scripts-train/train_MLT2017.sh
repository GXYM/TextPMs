#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_TextPMs.py --exp_name MLT2017 --net resnet50 --optim SGD --lr 0.01 --input_size 640 --batch_size 6 --gpu 0 --max_epoch 300 --num_workers 24 --viz --start_epoch 0 --save_freq 2
