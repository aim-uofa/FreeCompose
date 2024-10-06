import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import torch
import argparse
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

def dino_img_score(img1_path,img2_path):
    image1 = Image.open(img1_path)
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
    outputs1 = model(**inputs1)
    image_features1 = outputs1.last_hidden_state
    image_features1 = image_features1.mean(dim=1)

    image2 = Image.open(img2_path)
    with torch.no_grad():
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
    outputs2 = model(**inputs2)
    image_features2 = outputs2.last_hidden_state
    image_features2 = image_features2.mean(dim=1)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0],image_features2[0]).item()
    sim = (sim+1)/2
    
    return sim
    
def phase_args():
    parser = argparse.ArgumentParser(description="CLI for clip score")
    parser.add_argument("--gen_dir", type=str, default="output/metirc_v1/ours", help="path to generated images")
    parser.add_argument("--ori_dir", type=str, default="data/v1_80_out/images", help="path to original images")
    parser.add_argument("--foreground", type=bool, default=False, help="foreground or background")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = phase_args()
    gen_dir = args.gen_dir
    ori_dir = args.ori_dir

    scores = []
    for name in tqdm(os.listdir(gen_dir), total=len(os.listdir(gen_dir))):
        gen_path = os.path.join(gen_dir,name)
        if args.foreground:
            ori_path = os.path.join(ori_dir,'_'.join(name.split('_')[1:]).split('.')[0]+'.png')
        else:
            ori_path = os.path.join(ori_dir,name.split('_')[0]+'.png')
        score = dino_img_score(ori_path,gen_path)
        scores.append(score)

    if args.foreground:
        print("Foreground DINO score: ",sum(scores)/len(scores))
    else:
        print("Background DINO score: ",sum(scores)/len(scores))