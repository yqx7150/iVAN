import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage

from model.model import InvISPNet
from dataset.mri_dataset import mriDataset
from config.config import get_arguments

from tqdm import tqdm
import cv2
import imageio
from skimage.measure import compare_psnr, compare_ssim, compare_mse, shannon_entropy,compare_nrmse
from matplotlib import pyplot as plt
import math
import scipy.io as io


os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
os.system('rm tmp')

parser = get_arguments()
parser.add_argument("--ckpt", type=str, help="Checkpoint path.") 
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save results. ")
parser.add_argument("--root1", type=str, default="./data/T2_test", help="Output images. ")
parser.add_argument("--root2", type=str, default="./data/T1_test", help="Input images. ")
parser.add_argument("--root3", type=str, default="./data/PD_test", help="Another input images. ")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))

ckpt_allname = args.ckpt.split("/")[-1]

def save_img(img, img_path):
    img = np.clip(img*255,0,255)
    cv2.imwrite(img_path, img)

def save_img_color(img, img_path):
    img = np.clip(img*255,0,255)
    
    img_1 = img[:, :, :: -1]
    cv2.imwrite(img_path, img_1)

def main(args):
    # ======================================define the model============================================
    net = InvISPNet(channel_in=2, channel_out=2, block_num=8)
    device = torch.device("cuda:0")
    
    net.to(device)
    net.eval()
    # load the pretrained weight if there exists one
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))
    
    print("[INFO] Start data load and preprocessing") 

    Dataset = mriDataset(opt=args,root1=args.root1,root2=args.root2,root3=args.root3)
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)


    PSNR=[]
    PSNR_REV2=[]
    PSNR_REV3=[]
    SSIM=[]
    SSIM_REV2=[]
    SSIM_REV3=[]
    MSE=[]
    NMSE=[]

    print("[INFO] Start test...") 
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        step_time = time.time() 

        input, target_forward, input_target = sample_batched['input_img'].to(device), sample_batched['target_forward_img'].to(device), \
                            sample_batched['input_target_img'].to(device)

        input_file_name2 = sample_batched['input2_name'][0]
        input_file_name3 = sample_batched['input3_name'][0]
        target_file_name = sample_batched['target_forward_name'][0]


        with torch.no_grad():
            reconstruct_for = net(input)
            reconstruct_for = torch.clamp(reconstruct_for, 0, 1)

            reconstruct_rev = net(reconstruct_for, rev=True)

        pred_rev = reconstruct_rev.detach().permute(0,2,3,1).squeeze()  
        pred_rev = torch.clamp(pred_rev, 0, 1).cpu().numpy() 
        pred_for = reconstruct_for.detach().permute(0,2,3,1).squeeze().cpu().numpy()   
        target_forward_patch = target_forward.permute(0,2,3,1).squeeze().cpu().numpy()   
        target_rev_patch = input_target.permute(0,2,3,1).squeeze().cpu().numpy()  
        
        pred_for_mean = (pred_for[:,:,0]+pred_for[:,:,1])/2
        
        if args.task == '2to1':

            target_rev_2 = target_rev_patch[:,:,0]
            target_rev_3 = target_rev_patch[:,:,1]
            pred_rev_2 = pred_rev[:,:,0]
            pred_rev_3 = pred_rev[:,:,1]
            
        if args.task == '1to1':    
            target_rev_2 = (target_rev_patch[:,:,0]+target_rev_patch[:,:,1])/2
            target_rev_3 = (target_rev_patch[:,:,0]+target_rev_patch[:,:,1])/2
            pred_rev_2 = (pred_rev[:,:,0] +pred_rev[:,:,1])/2
            pred_rev_3 = (pred_rev[:,:,0] +pred_rev[:,:,1])/2

        psnr = compare_psnr( 255 * abs(target_forward_patch),255 * abs(pred_for_mean), data_range=255)
        psnr_rev_2 = compare_psnr( 255 * abs(target_rev_2),255 * abs(pred_rev_2), data_range=255)
        psnr_rev_3 = compare_psnr( 255 * abs(target_rev_3),255 * abs(pred_rev_3), data_range=255)
        ssim = compare_ssim(abs(target_forward_patch), abs(pred_for_mean), data_range=1)
        ssim_rev_2 = compare_ssim(abs(target_rev_2), abs(pred_rev_2), data_range=1)
        ssim_rev_3 = compare_ssim(abs(target_rev_3), abs(pred_rev_3), data_range=1)
        mse = compare_mse(target_forward_patch,pred_for_mean)

        nmse =  np.sum((pred_for_mean - target_forward_patch) ** 2.) / np.sum(target_forward_patch**2)

        PSNR.append(psnr)
        PSNR_REV2.append(psnr_rev_2)
        PSNR_REV3.append(psnr_rev_3)
        SSIM.append(ssim)
        SSIM_REV2.append(ssim_rev_2)
        SSIM_REV3.append(ssim_rev_3)
        MSE.append(mse)

        NMSE.append(nmse)

        save_path= 'exps/test/{}'.format(ckpt_allname)
        if not os.path.exists(save_path):
            os.makedirs(save_path+'/pred', exist_ok=True)
            os.makedirs(save_path+'/target', exist_ok=True)           
            os.makedirs(save_path+'/pred_rev_2', exist_ok=True)    
            os.makedirs(save_path+'/pred_rev_3', exist_ok=True)
            os.makedirs(save_path+'/target_rev_2', exist_ok=True)
            os.makedirs(save_path+'/target_rev_3', exist_ok=True)

        save_img(pred_for_mean, save_path+'/pred'+'/pred_'+target_file_name+'_{}.png'.format(ckpt_allname))
        save_img(target_forward_patch, save_path+'/target'+'/target_'+target_file_name+'_{}.png'.format(ckpt_allname))
        save_img(pred_rev_2, save_path+'/pred_rev_2'+'/pred_rev_'+input_file_name2+'_{}.png'.format(ckpt_allname))  
        save_img(pred_rev_3, save_path+'/pred_rev_3'+'/pred_rev_'+input_file_name3+'_{}.png'.format(ckpt_allname))
        save_img(target_rev_2, save_path+'/target_rev_2'+'/target_rev_'+input_file_name2+'_{}.png'.format(ckpt_allname))  
        save_img(target_rev_3, save_path+'/target_rev_3'+'/target_rev_'+input_file_name3+'_{}.png'.format(ckpt_allname)) 

        del reconstruct_for
        del reconstruct_rev

    ave_psnr = sum(PSNR) / len(PSNR)
    ave_psnr_rev2 = sum(PSNR_REV2) / len(PSNR_REV2)
    ave_psnr_rev3 = sum(PSNR_REV3) / len(PSNR_REV3)
    ave_ssim = sum(SSIM) / len(SSIM)
    ave_ssim_rev2 = sum(SSIM_REV2) / len(SSIM_REV2)
    ave_ssim_rev3 = sum(SSIM_REV3) / len(SSIM_REV3)
    ave_mse = sum(MSE) / len(MSE)
    
    ave_nmse = sum(NMSE) / len(NMSE)
    

    print('ave_psnr',ave_psnr)
    print('ave_psnr_rev2',ave_psnr_rev2)
    print('ave_psnr_rev3',ave_psnr_rev3)
    print('ave_ssim',ave_ssim)
    print('ave_ssim_rev2',ave_ssim_rev2)
    print('ave_ssim_rev3',ave_ssim_rev3)
    print('ave_mse',ave_mse)
    print('ave_nmse',ave_nmse)
        


if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)

