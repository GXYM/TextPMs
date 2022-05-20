#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 train_TextPMs.py --exp_name Totaltext --net resnet50 --optim SGD --lr 0.01 --input_size 800 --batch_size 4 --gpu 0 --max_epoch 300 --num_workers 24 --resume pretrained/totaltext_pretain/TextPMs_resnet50_300.pth --start_epoch 300 --viz
