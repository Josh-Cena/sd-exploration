import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import argparse
from create_timesteps import create_timesteps
import re
import time
import pandas as pd

device = "cuda"

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "tianweiy/DMD2"
ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
# Load model.
unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device))
pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_image(jobs: list[tuple[str, str, int]], seed=None):
    all_times = []
    for (prompt, scheduler, num_inference_steps) in jobs:
        if seed is not None:
            set_seed(seed)
        start = time.perf_counter()
        timesteps = create_timesteps(scheduler, num_inference_steps)
        image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0, timesteps=timesteps).images[0]
        end = time.perf_counter()
        all_times.append({ "prompt": prompt, "scheduler": scheduler, "num_inference_steps": num_inference_steps, "time": end - start })
        prompt_abstract = re.sub("[^0-9a-zA-Z]+", "_", "_".join(prompt.split(" ")[:4]))
        image.save(f"output/{seed}/{prompt_abstract}_{scheduler}_{num_inference_steps}.png")
    pd.DataFrame(all_times).to_csv(f"output/{seed}/times.csv", index=False)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt", type=str, required=True)
    argparser.add_argument("--scheduler", type=str, required=True, choices=["linear", "cosine", "exponential"])
    argparser.add_argument("--num-inference-steps", type=int, required=True)
    args = argparser.parse_args()
    generate_image([(args.prompt, args.scheduler, args.num_inference_steps)])
