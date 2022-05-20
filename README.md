# TextPMs
This is a Pytorch implementation of "Arbitrary Shape Text Detection via Segmentation with Probability Maps "  

![](https://github.com/GXYM/TextPMs/blob/master/vis/framework.png)

NOTE: This paper and project were completed in January 2020 and accepted by PAMI in May 2022. 

## Prerequisites  
  python 3.7;  
  PyTorch 1.2.0;   
  Numpy >=1.16;   
  CUDA >=10.2;  
  GCC >=9.0;
  *opencv-python < 4.5.0*     
  NVIDIA GPU(1080, 2080 or 3080);  

  NOTE: We tested the code in the environment of Arch Linux+Python3.7 with 1080, and  Arch Linux+Python3.9 with 2080. For other environments, the code may need to be adjusted slightly.


## Makefile

If “pse” is used, some cpp files need to be compiled

```
cd pse & make
```


## Dataset Links  
1. [CTW1500](https://drive.google.com/file/d/1A2s3FonXq4dHhD64A2NCWc8NQWMH2NFR/view?usp=sharing)   
2. [TD500](https://drive.google.com/file/d/1ByluLnyd8-Ltjo9AC-1m7omZnI-FA1u0/view?usp=sharing)  
3. [Total-Text](https://drive.google.com/file/d/17_7T_-2Bu3KSSg2OkXeCxj97TBsjvueC/view?usp=sharing) 

NOTE: The images of each dataset can be obtained from their official website.


## Training 
### Prepar dataset
We provide a simple example for each dataset in data, such as [Total-Text](https://github.com/GXYM/TextPMs/tree/master/data/total-text-mat), [CTW-1500](https://github.com/GXYM/TextPMs/tree/master/data/ctw1500), and [MLT-2017](https://github.com/GXYM/TextPMs/tree/master/data/MLT2017) ...


### Pre-training models
We provide some pre-tarining models on SynText and MLT-2017 [Baidu Drive](https://pan.baidu.com/s/1qHer8pXKUuXGRzwPrkH-dw) (download code: 07pb), [Google Drive](https://drive.google.com/file/d/1yuEO5fWKZJ9Sc0SA-qyDYZLHRKBlIe8a/view?usp=sharing)


### Models
 *  Total-Text model: [Baidu Drive](https://pan.baidu.com/s/1z5u8jEoKX2OxDvfRbO-ijw) (download code: ce36), [Google Drive](https://drive.google.com/file/d/12DQcfXf8mtAnBfRR3Trw2D1DfyV72Fmt/view?usp=sharing)
 *  CTW-1500 model: [Baidu Drive](https://pan.baidu.com/s/1o2Lwn5v6D4fhj_GiPEZ4yw) (download code: 7gov), [Google Drive](https://drive.google.com/file/d/1zT8EXrGpWIjegBZK8c4zM_x6MsmiA1q5/view?usp=sharing)
 *  MSRA-TD500 model: [Baidu Drive](https://pan.baidu.com/s/1pBNWnPG4YicGj8kiHuTFDg) (download code: yocp), [Google Drive](https://drive.google.com/file/d/1xOdCQDj2hXTKcpFR0LdmbpFutTKEF-cb/view?usp=sharing)
 *  ICDAR2017 model: [Baidu Drive](https://pan.baidu.com/s/1wOnoxRxt-bE0w9bvzQCnOw) (download code: eu1s), [Google Drive](https://drive.google.com/file/d/1JKVqjZAZs4mckhH7KiC7sNDqUw8Ib1Km/view?usp=sharing)
 
 NOTE: The model of each benchmark is pre-trained on MLT-2017; the trained model of MLT-2017 in pre-training models，so there is no link separately here.

 ### Runing the training scripts
We provide training scripts for each dataset in scripts-train, such as [Total-Text](https://github.com/GXYM/TextPMs/blob/master/scripts-train/train_Totaltxt.sh), [MLT-2017](https://github.com/GXYM/TextPMs/blob/master/scripts-train/train_MLT2017.sh), and [ArT](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/scripts-train) ...


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

### Demo
You can also run prediction on your own dataset without annotations. Here is an example:

``` 
#demo.sh
#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 python3 demo.py --net resnet18 --scale 4 --exp_name TD500 --checkepoch 1135 --test_size 640 960 --dis_threshold 0.35 --cls_threshold 0.9 --gpu 0 --viz --img_root /path/to/image 
```


### Evaluate the performance

Note that we provide some the protocols for benchmarks ([Total-Text](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/dataset/total_text), [CTW-1500](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/dataset/ctw1500), [MSRA-TD500](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/dataset/TD500), [ICDAR2015](https://github.com/GXYM/TextBPN-Plus-Plus/tree/main/dataset/icdar15)). The embedded evaluation protocol in the code are obtatined from the official protocols. You don't need to run these protocols alone, because our test code will automatically call these scripts, please refer to "[util/eval.py](https://github.com/GXYM/TextBPN-Plus-Plus/blob/main/util/eval.py)"




## Visualization
![](https://github.com/GXYM/TextPMs/blob/master/visual/img1.png](https://github.com/GXYM/TextPMs/blob/master/vis/img1.png)


## Citing the related works

Please cite the related works in your publications if it helps your research:

``` 
@article{Zhang2022PMs,
  title={Arbitrary Shape Text Detection via Segmentation with Probability Maps},
  author={Shi-Xue Zhang and Xiaobin Zhu and Lei Chen and Jie-Bo Hou and Xu-Cheng Yin},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2022},
  volume={PP}
}
  ``` 

## License  
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/TextPMs/blob/master/LICENSE.md) file for details


