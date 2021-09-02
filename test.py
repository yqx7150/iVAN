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
args = parser.parse_args()
print("Parsed arguments: {}".format(args))


ckpt_name = args.ckpt.split("/")[-1].split(".")[0]
ckpt_allname = args.ckpt.split("/")[-1]

def save_img(img, img_path):
    img = np.clip(img*255,0,255)
    cv2.imwrite(img_path, img)

def save_img_color(img, img_path):
    img = np.clip(img*255,0,255)
    
    img_1 = img[:, :, :: -1]
    cv2.imwrite(img_path, img_1)

def avgGradient(image):
    
    width = image.shape[1]
    width = width - 1
    heigt = image.shape[0]
    heigt = heigt - 1
    tmp = 0.0

    for i in range(width):
        for j in range(heigt):
            dx = float(image[i, j + 1]) - float(image[i, j])
            dy = float(image[i + 1, j]) - float(image[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds

    imageAG = tmp / (width * heigt)
    return round(imageAG,3)

def main(args):
    # ======================================define the model============================================
    net = InvISPNet(channel_in=4, channel_out=4, block_num=8)
    device = torch.device("cuda:0")
    
    net.to(device)
    net.eval()
    # load the pretrained weight if there exists one
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))
    
    print("[INFO] Start data load and preprocessing") 

    Dataset = mriDataset(opt=args,root1='./data/T1_T2/T2_test_mat',root2='./data/T1_T2/T1_test_mat',root3='./data/T1_T2/PD_test_mat')
    dataloader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    files=[]
    PSNR=[]
    PSNR_REV2=[]
    PSNR_REV3=[]
    SSIM=[]
    SSIM_REV2=[]
    SSIM_REV3=[]
    MSE=[]
    NRMSE=[]
    
    NMSE=[]
    #AG=[]
    #EN=[]
    
    NRMSE=[]
    print("[INFO] Start test...") 
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        step_time = time.time() 
        #input 0,1
        input, target_1, target_rev = sample_batched['input_img'].to(device), sample_batched['target_1'].to(device), \
                            sample_batched['input_target_img'].to(device)

        input_file_name2 = sample_batched['input_file_name2'][0]
        input_file_name3 = sample_batched['input_file_name3'][0]
        target_file_name = sample_batched['target_file_name'][0]
        print(input_file_name2)
        print(input_file_name3)
        print(target_file_name)

        with torch.no_grad():
            reconstruct_1 = net(input)
            reconstruct_1 = torch.clamp(reconstruct_1, 0, 1)

            reconstruct_rev = net(reconstruct_1, rev=True)

        pred_rev = reconstruct_rev.detach().permute(0,2,3,1).squeeze()  
        pred_rev = torch.clamp(pred_rev, 0, 1).cpu().numpy() 
        pred_1 = reconstruct_1.detach().permute(0,2,3,1).squeeze().cpu().numpy()   
        target_1_patch = target_1.permute(0,2,3,1).squeeze().cpu().numpy()   
        target_rev_patch = target_rev.permute(0,2,3,1).squeeze().cpu().numpy()  


        pred_1_mean = (pred_1[:,:,0]+pred_1[:,:,1]+pred_1[:,:,2]+pred_1[:,:,3])/4  
        
        target_rev_2 = (target_rev_patch[:,:,0]+target_rev_patch[:,:,2])/2
        target_rev_3 = (target_rev_patch[:,:,1]+target_rev_patch[:,:,3])/2
        pred_rev_2 = (pred_rev[:,:,0]+pred_rev[:,:,2])/2
        pred_rev_3 = (pred_rev[:,:,1]+pred_rev[:,:,3])/2
        

        
        print('pred_1_mean',np.max(pred_1_mean),np.min(pred_1_mean),pred_1_mean.shape)
        print('target_1_patch',np.max(target_1_patch),np.min(target_1_patch),target_1_patch.shape)
        #print('pred_rev_2',np.max(pred_rev_2),np.min(pred_rev_2),pred_rev_2.shape)
        #print('target_rev_2',np.max(target_rev_2),np.min(target_rev_2),target_rev_2.shape)
        #print('pred_rev_3',np.max(pred_rev_3),np.min(pred_rev_3),pred_rev_3.shape)
        #print('target_rev_3',np.max(target_rev_3),np.min(target_rev_3),target_rev_3.shape)

        assert 0

        psnr = compare_psnr( 255 * abs(target_1_patch),255 * abs(pred_1_mean), data_range=255)
        psnr_rev_2 = compare_psnr( 255 * abs(target_rev_2),255 * abs(pred_rev_2), data_range=255)
        psnr_rev_3 = compare_psnr( 255 * abs(target_rev_3),255 * abs(pred_rev_3), data_range=255)
        ssim = compare_ssim(abs(target_1_patch), abs(pred_1_mean), data_range=1,multichannel=True)
        ssim_rev_2 = compare_ssim(abs(target_rev_2), abs(pred_rev_2), data_range=1,multichannel=True)
        ssim_rev_3 = compare_ssim(abs(target_rev_3), abs(pred_rev_3), data_range=1,multichannel=True)
        mse = compare_mse(target_1_patch,pred_1_mean)
        
        nmse =  np.sum((pred_1_mean - target_1_patch) ** 2.) / np.sum(target_1_patch**2)
        
        nrmse = compare_nrmse(target_1_patch,pred_1_mean,norm_type='euclidean')
        #ag = avgGradient(pred_1_mean)
        #en = shannon_entropy(pred_1_mean,base=2)
        #print('===============================psnr',psnr)
        #print('===============================psnr_rev_2',psnr_rev_2)
        #print('===============================psnr_rev_3',psnr_rev_3)
        #print('===============================ssim_rev_2',ssim_rev_2)
        #print('===============================ssim_rev_3',ssim_rev_3)

        files.append(target_file_name)
        PSNR.append(psnr)
        PSNR_REV2.append(psnr_rev_2)
        PSNR_REV3.append(psnr_rev_3)
        SSIM.append(ssim)
        SSIM_REV2.append(ssim_rev_2)
        SSIM_REV3.append(ssim_rev_3)
        MSE.append(mse)
        
        NMSE.append(nmse)
        
        NRMSE.append(nrmse)
        #AG.append(ag)
        #EN.append(en)

        save_path= 'exps/test/{}'.format(ckpt_allname)
        if not os.path.exists(save_path):
            os.makedirs(save_path+'/pred')
            os.makedirs(save_path+'/pred_mat')
            
            os.makedirs(save_path+'/target')
            os.makedirs(save_path+'/target_mat')
            
            os.makedirs(save_path+'/pred_rev_2')
            os.makedirs(save_path+'/pred_rev_2_mat')
            
            os.makedirs(save_path+'/pred_rev_3')
            os.makedirs(save_path+'/pred_rev_3_mat')
            
            os.makedirs(save_path+'/target_rev_2')
            os.makedirs(save_path+'/target_rev_2_mat')
            
            os.makedirs(save_path+'/target_rev_3')
            os.makedirs(save_path+'/target_rev_3_mat')

        save_img(pred_1_mean, save_path+'/pred'+'/pred_'+target_file_name+'_{}.png'.format(ckpt_allname))
        io.savemat(save_path+'/pred_mat'+'/pred_'+target_file_name+'_{}.mat'.format(ckpt_allname),{'data':pred_1_mean})
        
        save_img(target_1_patch, save_path+'/target'+'/target_'+target_file_name+'_{}.png'.format(ckpt_allname))
        io.savemat(save_path+'/target_mat'+'/target_'+target_file_name+'_{}.mat'.format(ckpt_allname),{'data':target_1_patch})
        
        save_img(pred_rev_2, save_path+'/pred_rev_2'+'/pred_rev_'+input_file_name2+'_{}.png'.format(ckpt_allname))
        io.savemat(save_path+'/pred_rev_2_mat'+'/pred_rev_'+input_file_name2+'_{}.mat'.format(ckpt_allname),{'data':pred_rev_2})
        
        save_img(pred_rev_3, save_path+'/pred_rev_3'+'/pred_rev_'+input_file_name3+'_{}.png'.format(ckpt_allname))
        io.savemat(save_path+'/pred_rev_3_mat'+'/pred_rev_'+input_file_name3+'_{}.mat'.format(ckpt_allname),{'data':pred_rev_3})
        
        save_img(target_rev_2, save_path+'/target_rev_2'+'/target_rev_'+input_file_name2+'_{}.png'.format(ckpt_allname))
        io.savemat(save_path+'/target_rev_2_mat'+'/target_rev_'+input_file_name2+'_{}.mat'.format(ckpt_allname),{'data':target_rev_2})
        
        save_img(target_rev_3, save_path+'/target_rev_3'+'/target_rev_'+input_file_name3+'_{}.png'.format(ckpt_allname))
        io.savemat(save_path+'/target_rev_3_mat'+'/target_rev_'+input_file_name3+'_{}.mat'.format(ckpt_allname),{'data':target_rev_3})
        
        #plt.subplot(2, 2, 1)
        #plt.imshow(255 * abs(target_rev_patch_mean), cmap='gray')
        #plt.title("target_(T1+T2)")
        #plt.subplot(2, 2, 2)
        #plt.imshow(255 * abs(pred_T2_mean), cmap='gray')
        #plt.title("pred_(T1+T2)")
        #plt.subplot(2, 2, 3)
        #plt.imshow(255 * abs(target_1_patch), cmap='gray')
        #plt.title("target_pet")
        #plt.subplot(2, 2, 4)
        #plt.imshow(255 * abs(pred_1), cmap='gray')
        #plt.title("pred_pet")
        # plt.ion()
        # plt.pause(1.5)
        # plt.close()
        #plt.show()

        del reconstruct_1
        del reconstruct_rev

    ave_psnr = sum(PSNR) / len(PSNR)
    ave_psnr_rev2 = sum(PSNR_REV2) / len(PSNR_REV2)
    ave_psnr_rev3 = sum(PSNR_REV3) / len(PSNR_REV3)
    ave_ssim = sum(SSIM) / len(SSIM)
    ave_ssim_rev2 = sum(SSIM_REV2) / len(SSIM_REV2)
    ave_ssim_rev3 = sum(SSIM_REV3) / len(SSIM_REV3)
    ave_mse = sum(MSE) / len(MSE)
    
    ave_nmse = sum(NMSE) / len(NMSE)
    
    ave_nrmse = sum(NRMSE) / len(NRMSE)
    #ave_ag = sum(AG) / len(AG)
    #ave_en = sum(EN) / len(EN)
    with open('results_test.txt', 'a+') as f:
        f.write('\n'*3)
        f.write(ckpt_allname+'\n')
        f.write(str(files)+'\n')
            
        
        f.write(str(len(PSNR))+'\n')
        f.write(str(PSNR)+'\n')
        f.write('ave_psnr:'+str(ave_psnr)+'\n')
        f.write(str(len(PSNR_REV2))+'\n')
        f.write(str(PSNR_REV2)+'\n')
        f.write('ave_psnr_rev2:'+str(ave_psnr_rev2)+'\n')
        f.write(str(len(PSNR_REV3))+'\n')
        f.write(str(PSNR_REV3)+'\n')
        f.write('ave_psnr_rev3:'+str(ave_psnr_rev3)+'\n')
        
        
        f.write(str(len(SSIM))+'\n')
        f.write(str(SSIM)+'\n')     
        f.write('ave_ssim:'+str(ave_ssim)+'\n')
        f.write(str(len(SSIM_REV2))+'\n')
        f.write(str(SSIM_REV2)+'\n')
        f.write('ave_ssim_rev2:'+str(ave_ssim_rev2)+'\n')
        f.write(str(len(SSIM_REV3))+'\n')
        f.write(str(SSIM_REV3)+'\n')
        f.write('ave_ssim_rev3:'+str(ave_ssim_rev3)+'\n')
        
        f.write(str(len(MSE))+'\n')
        f.write(str(MSE)+'\n')
        f.write('ave_mse:'+str(ave_mse)+'\n')
        
        
        f.write(str(len(NMSE))+'\n')
        f.write(str(NMSE)+'\n')
        f.write('ave_nmse:'+str(ave_nmse)+'\n')
        
        f.write(str(len(NRMSE))+'\n')
        f.write(str(NRMSE)+'\n')
        f.write('ave_nrmse:'+str(ave_nrmse)+'\n')
        
        #f.write(str(len(AG))+'\n')
        #f.write(str(AG)+'\n')
        #f.write('ave_ag:'+str(ave_ag)+'\n')
        
        #f.write(str(len(EN))+'\n')
        #f.write(str(EN)+'\n')
        #f.write('ave_en:'+str(ave_en)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)

