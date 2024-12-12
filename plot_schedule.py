import numpy as np
import matplotlib.pyplot as plt
from create_timesteps import create_timesteps

schedules = [
    "linear",
    "cosine",
    "exponential"
]

num_inference_steps = [
    3,
    4,
    8,
    15,
]

fig, axs = plt.subplots(len(schedules), len(num_inference_steps), figsize=(15, 15))

for schedule in schedules:
    for num_inference_step in num_inference_steps:
        step_range = create_timesteps(schedule, num_inference_step)
        ax = axs[schedules.index(schedule), num_inference_steps.index(num_inference_step)]
        ax.scatter(range(num_inference_step), step_range)
        ax.set_title(f"{schedule}, {num_inference_step} steps")

fig.savefig("schedule.png")
