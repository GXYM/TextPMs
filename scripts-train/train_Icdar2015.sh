#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_TextPMs.py --exp_name Icdar2015 --net resnet50 --optim SGD --lr 0.01 --input_size 640 --batch_size 6 --gpu 0 --max_epoch 300 --num_workers 24 --resume pretrained/MLT2017/TextPMs_resnet50_64.pth --start_epoch 600 --viz
