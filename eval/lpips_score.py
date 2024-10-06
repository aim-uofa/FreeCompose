import os
import lpips
import argparse
import torch
from tqdm import tqdm

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def phase_args():
    parser = argparse.ArgumentParser(description="CLI for clip score")
    parser.add_argument("--gen_dir", type=str, default="output/metirc_v1/ours", help="path to generated images")
    parser.add_argument("--ori_dir", type=str, default="data/v1_80_out/images", help="path to original images")
    args = parser.parse_args()
    return args

def alex_lpips_score(img1_path,img2_path):
    img1 = lpips.im2tensor(lpips.load_image(img1_path))
    img2 = lpips.im2tensor(lpips.load_image(img2_path))
    return loss_fn_alex.forward(img1, img2).item()

def vgg_lpips_score(img1_path,img2_path):
    img1 = lpips.im2tensor(lpips.load_image(img1_path))
    img2 = lpips.im2tensor(lpips.load_image(img2_path))
    return loss_fn_vgg.forward(img1, img2).item()

if __name__ == "__main__":
    args = phase_args()
    gen_dir = args.gen_dir
    ori_dir = args.ori_dir

    img_names = sorted([i for i in os.listdir(ori_dir)])
    alex_scores = []
    vgg_scores = []
    for img_name in tqdm(img_names, total=len(img_names)):
        ori_path = os.path.join(ori_dir, img_name)
        gen_path = os.path.join(gen_dir, img_name)
        alex_scores.append(alex_lpips_score(ori_path, gen_path))
        vgg_scores.append(vgg_lpips_score(ori_path, gen_path))
        
    print("Alex LPIPS score: ", sum(alex_scores) / len(alex_scores))
    print("VGG LPIPS score: ", sum(vgg_scores) / len(vgg_scores))