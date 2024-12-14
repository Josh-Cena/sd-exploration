"""
Prepare images for FID/MSE/etc. scores.
"""

import os
import re
from pathlib import Path
from PIL import Image

def reorder_images(src_dir: str, dst_dir: str):
    for seed in os.listdir(src_dir):
        for file in os.listdir(f"{src_dir}/{seed}"):
            if not file.endswith(".png"):
                continue
            match = re.match(r"(.*)_(\w+_\d+)\.png", file)
            src = f"{src_dir}/{seed}/{file}"
            dst = f"{dst_dir}/{match.group(2)}/{match.group(1)}-{seed}.png"
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(src)
            img = img.resize((64, 64))
            img.save(dst)

if __name__ == "__main__":
    reorder_images("output", "output-alt")
