# calculate ssim, mse, fmse, psnr score of two dirs' images

import os
import cv2
import numpy as np
import argparse
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from prettytable import PrettyTable

def phase_args():
    parser = argparse.ArgumentParser(description="CLI for clip score")
    parser.add_argument("--gen_dir", type=str, default="output/metirc_v1/ours", help="path to generated images")
    parser.add_argument("--ori_dir", type=str, default="data/v1_80_out/images", help="path to original images")
    parser.add_argument("--mask_dir", type=str, default="data/v1_80_out/masks", help="path to mask images")
    parser.add_argument("--task", type=str, default="com", help="task to evaluate")
    args = parser.parse_args()
    return args

def ssim_score(img1_path,img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    r_ss = ssim(img1[:,:,0], img2[:,:,0])
    g_ss = ssim(img1[:,:,1], img2[:,:,1])
    b_ss = ssim(img1[:,:,2], img2[:,:,2])
    # average ssim of 3 channels
    return (r_ss+g_ss+b_ss) / 3

def grey_ssim_score(img1_path,img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    return ssim(img1, img2)

def mse_score(img1_path,img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    return mean_squared_error(img1, img2)

def grey_mse_score(img1_path,img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    return mean_squared_error(img1, img2)

def fmse_score(img1_path,img2_path,mask_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    mask = cv2.imread(mask_path)
    # if gray mask, repeat to 3 channels
    if len(mask.shape) == 2:
        mask = np.repeat(mask[:,:,np.newaxis],3,axis=2)
    # either 0 or 255, normalize to 0 or 1 for
    mask = np.where(mask>0,1,0)
    return mean_squared_error(img1*mask, img2*mask)

def grey_fmse_score(img1_path,img2_path,mask_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask>0,1,0)
    return mean_squared_error(img1*mask, img2*mask)

def psnr_score(img1_path,img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    return psnr(img1, img2)

def grey_psnr_score(img1_path,img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    return psnr(img1, img2)

if __name__ == "__main__":
    args = phase_args()
    gen_dir = args.gen_dir
    ori_dir = args.ori_dir
    mask_dir = args.mask_dir

    ssim_scores = []
    mse_scores = []
    fmse_scores = []
    psnr_scores = []
    
    grey_ssim_scores = []
    grey_mse_scores = []
    grey_fmse_scores = []
    grey_psnr_scores = []
    
    for name in tqdm(os.listdir(gen_dir), total=len(os.listdir(gen_dir))):
        gen_path = os.path.join(gen_dir,name)
        if args.task == 'com':
            ori_path = os.path.join(ori_dir,name.split('_')[0]+'.png')
            mask_path = os.path.join(mask_dir,name.split('_')[0]+'.png')
        elif args.task == 'rml':
            ori_path = os.path.join(ori_dir,name)
            mask_path = os.path.join(mask_dir,name)
        
        ss = ssim_score(ori_path,gen_path)
        mse = mse_score(ori_path,gen_path)
        fmse = fmse_score(ori_path,gen_path,mask_path)
        p = psnr_score(ori_path,gen_path)
        
        grey_ss = grey_ssim_score(ori_path,gen_path)
        grey_mse = grey_mse_score(ori_path,gen_path)
        grey_fmse = grey_fmse_score(ori_path,gen_path,mask_path)
        grey_p = grey_psnr_score(ori_path,gen_path)
        
        ssim_scores.append(ss)
        mse_scores.append(mse)
        fmse_scores.append(fmse)
        psnr_scores.append(p)
        
        grey_ssim_scores.append(grey_ss)
        grey_mse_scores.append(grey_mse)
        grey_fmse_scores.append(grey_fmse)
        grey_psnr_scores.append(grey_p)

    # print results as 3 row 5 columns table
    table = PrettyTable([gen_dir.split('/')[-1], 'SSIM', 'MSE', 'FMSE', 'PSNR'])
    table.add_row(['RGB', np.mean(ssim_scores), np.mean(mse_scores), np.mean(fmse_scores), np.mean(psnr_scores)])
    table.add_row(['Grey', np.mean(grey_ssim_scores), np.mean(grey_mse_scores), np.mean(grey_fmse_scores), np.mean(grey_psnr_scores)])
    print(table)
