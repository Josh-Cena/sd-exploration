"""
Prepare images for FID/MSE/etc. scores.
"""

import os
import re
from pathlib import Path
from PIL import Image

for seed in os.listdir("output"):
    for file in os.listdir(f"output/{seed}"):
        if not file.endswith(".png"):
            continue
        match = re.match(r"(.*)_(\w+_\d+)\.png", file)
        src = f"output/{seed}/{file}"
        dst = f"output-alt/{match.group(2)}/{match.group(1)}-{seed}.png"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src)
        img = img.resize((64, 64))
        img.save(dst)
