# Variable Augmented Network for Invertible Modality Synthesis-Fusion
The Code is created based on the method described in the following paper:
Variable Augmented Network for Invertible Modality Synthesis-Fusion
Author: Y. Wang, R. Liu, Z. Li, C. Yang.   
Date :   
Version : 1.0   
The code and the algorithm are for non-comercial use only.   
Copyright 2021, Department of Electronic Information Engineering, Nanchang University.   

## Optional parameters:  
weight: Weight for forward loss

## Visual illustration of the invertible medical image synthesis and fusion in variable augmentation manner
 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig2.jpg"> </div>
 
## The training pipeline of iVAN
 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig3.png"> </div>
 
## Two visualization results of synthesizing from T1 to T2
 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig6.jpg"> </div>
 
## Three fusion results of T2-weighted MR and CT images
 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig9.jpg"> </div>

 
## Train

Prepare your own datasets for VAN

You need to create at least two modality medical images from domain A /data/A and from domain B /data/B. Then you can train the model with the dataset flag --root1 './data/A' --root2 './data/B'. Optionally, you can create hold-out test datasets at ./data/A_test and ./data/B_test to test your model.

##  1to1
python train.py --task=1to1 --out_path="./exps/"

##  many to 1
python train.py --task=2to1 --out_path="./exps/"

##  resume training:
To fine-tune a pre-trained model, or resume the previous training, use the --resume flag


## Test

python test.py --task=test --out_path="./exps/" --ckpt="./exps/train/checkpoint/latest.pth"
