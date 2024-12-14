import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def make_demos(seed: int, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    images_by_prompt: dict[str, list[tuple[str, int, str]]] = {}
    for filename in os.listdir(f"output/{seed}"):
        if filename.endswith(".png"):
            match = re.match(r"^(.*?)_(linear|exponential|cosine)_(\d+)\.png$", filename)
            if match:
                prompt, scheduler, timesteps = match.groups()
                timesteps = int(timesteps)
                if prompt not in images_by_prompt:
                    images_by_prompt[prompt] = []
                images_by_prompt[prompt].append((scheduler, timesteps, filename))

    for prompt, images in images_by_prompt.items():
        schedulers = ["linear", "exponential", "cosine"]
        timesteps = sorted(set(img[1] for img in images))
        grid = {scheduler: {ts: None for ts in timesteps} for scheduler in schedulers}
        
        for scheduler, ts, filename in images:
            grid[scheduler][ts] = Image.open(f"output/{seed}/{filename}")

        fig, axes = plt.subplots(len(schedulers), len(timesteps), figsize=(15, 10))

        for i, scheduler in enumerate(schedulers):
            for j, ts in enumerate(timesteps):
                ax = axes[i, j]
                if grid[scheduler][ts]:
                    ax.imshow(grid[scheduler][ts])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_xlabel(f"{ts}", fontsize=10)
                    ax.xaxis.set_label_position("top")
                if j == 0:
                    ax.set_ylabel(f"{scheduler}", fontsize=10)
                    ax.yaxis.set_label_position("left")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prompt}.png")
        plt.close()

if __name__ == "__main__":
    make_demos(431, "plots/output_demos")
