# TextPMs
Arbitrary Shape Text Detection in Scene Images  
## 1.Prerequisites  
**python 3.7**;  
**PyTorch 1.2.0**;   
**Numpy >=1.16**;   
**CUDA 10.2**;  
**GCC >=9.0**;   
**NVIDIA GPU(with 10G or larger GPU memory for inference)**;   
## 2.Models
 *   [Total-Text model](https://drive.google.com/open?id=1cyAW7X4LESCJV6pEcSWw3BnXOnZSSPPC) pretrained on ICDAR2017-MLT.
 *  [CTW-1500 model](https://drive.google.com/open?id=1cyAW7X4LESCJV6pEcSWw3BnXOnZSSPPC) pretrained on ICDAR2017-MLT.
 *  [MSRA-TD500 model](https://drive.google.com/open?id=1WKFJsotug9qeuMxqnmgBbMPDR6CaujsM) pretrained on ICDAR2017-MLT.  

## 3.Running Eval
* **Preparation**  
2. put your test images in "data/TD500/Test" or data/ctw1500/test/text_image
3. put the pretrained model into ["model/TD500/"](https://github.com/anoycode22/DRRG/tree/master/model/TD500) or ["model/Ctw1500"](https://github.com/anoycode22/DRRG/tree/master/model/Ctw1500)
4. cd ./csrc and make
5. cd ./nmslib/lanms and make
