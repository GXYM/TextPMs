#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 python3 demo.py --net resnet50 --exp_name Totaltext --checkepoch 250 --test_size 640 1024 --threshold 0.4 --score_i 0.7 --recover watershed --gpu 0 --img_root ./demo  --viz
