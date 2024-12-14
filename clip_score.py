import torch
import clip
from PIL import Image
from all_jobs import jobs
import re
import pandas as pd
import argparse

def compute_clip_score(seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    similarities = []
    for (prompt, scheduler, num_inference_steps) in jobs:
        text_prompts = [prompt]
        prompt_abstract = re.sub("[^0-9a-zA-Z]+", "_", "_".join(prompt.split(" ")[:4]))
        images = [Image.open(f"output/{seed}/{prompt_abstract}_{scheduler}_{num_inference_steps}.png")]

        image_features = torch.stack([preprocess(img).to(device) for img in images])
        image_features = model.encode_image(image_features)

        text_features = model.encode_text(clip.tokenize(text_prompts).to(device))

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).mean().item()
        similarities.append({ "prompt": prompt, "scheduler": scheduler, "num_inference_steps": num_inference_steps, "similarity": similarity })

    pd.DataFrame(similarities).to_csv(f"output/{seed}/similarities.csv", index=False)

if __name__ == "__main__":
    # Work around CUDA memory limits.
    # Run this script with the following command:
    # for dir in output/*; do
    #   if [ -d "$dir" ]; then
    #     suffix=${dir#output/}
    #     python clip_score.py --seed "$suffix"
    #   fi
    # done
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, required=True)
    args = argparser.parse_args()
    compute_clip_score(args.seed)
