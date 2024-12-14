import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

prompts = [
    "A_photo_of_llama",
    "a_shiba_inu_wearing",
    "A_train_ride_in",
    "a_young_girl_playing",
]

scores: dict[str, dict[str, list[float]]] = {}

for group in os.listdir("output-alt"):
    scheduler, steps = group.split("_")
    for file in os.listdir(f"output-alt/{group}"):
        if not file.endswith(".png"):
            continue
        match = re.match(r"(.*)-(\d+)\.png", file)
        src = f"output-alt/{group}/{file}"
        img = Image.open(src)
        img = np.array(img)
        truth: Image.Image = Image.open(f"prompt{prompts.index(match.group(1)) + 1}-real.png")
        truth = truth.resize((64, 64))
        truth = truth.convert("RGB")
        truth = np.array(truth)
        mse = np.mean((img - truth) ** 2)
        scores.setdefault(scheduler, {}).setdefault(steps, []).append(mse)

fig, ax = plt.subplots()
for scheduler, steps in sorted(scores.items(), key=lambda x: x[0]):
    curve = [(int(k), np.mean(v)) for k, v in steps.items()]
    curve.sort()
    x, y = zip(*curve)
    ax.plot(x, y, label=scheduler)
ax.legend()
ax.set_xlabel("Steps")
ax.set_ylabel("MSE")
fig.tight_layout()
fig.savefig("mse_score.png")
