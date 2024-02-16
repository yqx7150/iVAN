# Variable Augmented Network for Invertible Modality Synthesis-Fusion    
The Code is created based on the method described in the following paper:     
Variable Augmented Network for Invertible Modality Synthesis and Fusion    
Y. Wang, R. Liu, Z. Li, S. Wang, C. Yang, Q. Liu   
IEEE Journal of Biomedical and Health Informatics
Page: 2898 - 2909, Volume: 27 Issue: 6, 2023.
https://ieeexplore.ieee.org/abstract/document/10070774               
Date : Sep. 1, 2021  
Version : 1.0   
The code and the algorithm are for non-comercial use only.   
Copyright 2021, Department of Electronic Information Engineering, Nanchang University.   

## Optional parameters:  
weight: Weight for forward loss.   
epoch: Specifies number of iterations.


## Visual illustration of the invertible medical image synthesis and fusion in variable augmentation manner
 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig2.jpg"> </div>
 
## The training pipeline of iVAN
 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig3.png"> </div>
 
## Two visualization results of synthesizing from T1 to T2
 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig6.jpg"> </div>
 
## Three fusion results of T2-weighted MR and CT images
 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig9.jpg"> </div>

 
# Train

Prepare your own datasets for VAN

You need to create at least two modality medical images from domain A /data/A and from domain B /data/B. Then you can train the model with the dataset flag --root1 './data/A' --root2 './data/B'. Optionally, you can create hold-out test datasets at ./data/A_test and ./data/B_test to test your model.

##  1to1
python train.py --task=1to1 --out_path="./exps/"

##  many to 1
python train.py --task=2to1 --out_path="./exps/"

##  resume training:
To fine-tune a pre-trained model, or resume the previous training, use the --resume flag


# Test

python test.py --task=2to1 --out_path="./exps/" --ckpt="./exps/2to1/checkpoint/latest.pth"

python test.py --task=1to1 --out_path="./exps/" --ckpt="./exps/1to1/checkpoint/latest.pth"

# Acknowledgement
The code is based on [yzxing87/Invertible-ISP](https://github.com/yzxing87/Invertible-ISP)


### Other Related Projects

 * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)             
  
  * Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)  [<font size=5>**[Paper]**</font>](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221205?viewType=HTML)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VA-IDN)    

 * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    

  * Synthetic CT Generation via Invertible Network for All-digital Brain PET Attenuation Correction  [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2310.01885)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)        

  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   
