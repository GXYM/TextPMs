# TextPMs
Arbitrary Shape Text Detection in Scene Images  
## 1.Prerequisites  
**python 3.7**;  
**PyTorch 1.2.0**;   
**Numpy >=1.16**;   
**CUDA 10.2**;  
**GCC >=9.0**;   
**NVIDIA GPU(with 10G or larger GPU memory for inference)**;  

# 2.Dataset Links  
1. [CTW1500](https://drive.google.com/file/d/1A2s3FonXq4dHhD64A2NCWc8NQWMH2NFR/view?usp=sharing)   
2. [TD500](https://drive.google.com/file/d/1ByluLnyd8-Ltjo9AC-1m7omZnI-FA1u0/view?usp=sharing)  
3. [Total-Text](https://drive.google.com/file/d/17_7T_-2Bu3KSSg2OkXeCxj97TBsjvueC/view?usp=sharing) 

## 3.Models
 *  [Total-Text model]() (pretrained on ICDAR2017-MLT)
 *  [CTW-1500 model]() (pretrained on ICDAR2017-MLT)
 *  [MSRA-TD500 model]() (pretrained on ICDAR2017-MLT)  

## 4.Running Evaluation
run:  
```
sh eval.sh
```
The details in a are as follows:  
```
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

```  
Note: Be sure to modify the test_size in [util/option.py](https://github.com/GXYM/TextPMs/blob/master/util/option.py) 
# 5.Visualization
![Visualization ](https://github.com/GXYM/TextPMs/blob/master/visual/img1.png)
# 6.License  
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/TextPMs/blob/master/LICENSE.md) file for details


