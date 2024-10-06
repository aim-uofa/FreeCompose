import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
# from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
from tqdm import tqdm
import argparse

# Load the CLIP model
model_ID = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_ID).to("cuda")

preprocess = CLIPImageProcessor.from_pretrained(model_ID)


# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)
    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")
    # Return the preprocessed image
    return image


def clip_img_score (img1_path,img2_path):
    # Load the two images and preprocess them for CLIP
    image_a = load_and_preprocess_image(img1_path)["pixel_values"].to("cuda")
    image_b = load_and_preprocess_image(img2_path)["pixel_values"].to("cuda")

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    return similarity_score.item()

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
        score = clip_img_score(ori_path,gen_path)
        scores.append(score)

    if args.foreground:
        print("Foreground CLIP score: ",sum(scores)/len(scores))
    else:
        print("Background CLIP score: ",sum(scores)/len(scores))