#!/bin/bash

##################### Total-Text ###################################
# threshold=0.3, score_i=0.65; test_size=(256,1024)--util/option
CUDA_LAUNCH_BLOCKING=1 python eval_pms.py --exp_name Totaltext --checkepoch 550 --threshold 0.3 --score_i 0.65 --min_area 300 --voting false --gpu 0

# threshold=0.325; test_size=(256,1024)
#CUDA_LAUNCH_BLOCKING=1 python eval_pms.py --exp_name Totaltext --checkepoch 550 --threshold 0.325 --min_area 300 --voting true --gpu 0

###################### CTW-1500 ####################################
# threshold=0.3, score_i=0.65;test_size=(512,1024)--util/option
#CUDA_LAUNCH_BLOCKING=1 python eval_pms.py --exp_name Ctw1500 --checkepoch 480 --threshold 0.3 --score_i 0.65 --min_area 300 --voting false --gpu 0

# threshold=0.365;test_size=(512,102I4)--util/option
#CUDA_LAUNCH_BLOCKING=1 python eval_pms.py --exp_name Ctw1500 --checkepoch 480 --threshold 0.365 --min_area 300  --voting  true --gpu 0


#################### MSRA-TD500 ######################################
# threshold=0.305, score_i=0.8; test_size=(0,832)--util/option
#CUDA_LAUNCH_BLOCKING=1 python eval_pms.py --exp_name TD500 --checkepoch 125 --threshold 0.305 --score_i 0.8 --min_area 150  --voting false --gpu 0

# threshold=0.305; test_size=(0,832)--util/option
#CUDA_LAUNCH_BLOCKING=1 python eval_pms.py --exp_name TD500 --checkepoch 125 --threshold 0.305 --min_area 150  --voting true --gpu 0
