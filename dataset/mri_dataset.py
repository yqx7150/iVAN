from __future__ import print_function, division
import os, random, time

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import rawpy
from glob import glob
from PIL import Image as PILImage
import numbers
from scipy.misc import imread
import scipy.io as io
import cv2


class mriDataset(Dataset):
    def __init__(self, opt,root1,root2,root3):

        input_2 = np.array([root2 +"/"+ x  for x in os.listdir(root2)])
        target_forward = np.array([root1 +"/"+ x  for x in os.listdir(root1)])
        input_3 = np.array([root3 +"/"+ x  for x in os.listdir(root3)])

        assert len(input_2) == len(target_forward) == len(input_3)
        self.data = {'input_2':input_2, 'target_forward':target_forward,'input_3':input_3}

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)

    def __len__(self):
        return len(self.data['input_2'])

    def __getitem__(self, idx):
        self.data['input_2'].sort()
        self.data['target_forward'].sort()
        self.data['input_3'].sort()
        input_2_path = self.data['input_2'][idx]
        target_forward_path = self.data['target_forward'][idx]
        input_3_path = self.data['input_3'][idx]

        
        assert (input_2_path.split('T1_')[-1]) == (target_forward_path.split('T2_')[-1]) == (input_3_path.split('PD_')[-1])

        input_2_data = io.loadmat(input_2_path)['data']
        target_forward_data = io.loadmat(target_forward_path)['data']
        input_3_data = io.loadmat(input_3_path)['data']

        h,w = input_2_data.shape

        target_forward_img = np.expand_dims(target_forward_data, 2) 

        input_2_img = np.expand_dims(input_2_data, 2)  
        input_3_img = np.expand_dims(input_3_data, 2)
        input_img = np.zeros((h,w,2))
        input_img[:,:,0] = input_2_img[:,:,0]
        input_img[:,:,1] = input_3_img[:,:,0]

        input_target_img = input_img.copy()

        input_img = self.np2tensor(input_img).float()
        target_forward_img = self.np2tensor(target_forward_img).float()
        input_target_img = self.np2tensor(input_target_img).float()

        sample = {'input_img':input_img, 'target_forward_img':target_forward_img, 'input_target_img':input_target_img,
                    'input2_name':input_2_path.split("/")[-1].split(".")[0],'input3_name':input_3_path.split("/")[-1].split(".")[0],'target_forward_name':target_forward_path.split("/")[-1].split(".")[0]}
        return sample


