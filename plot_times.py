import pandas as pd
import matplotlib.pyplot as plt
import os

times = [f"{x}/times.csv" for x in os.listdir("output")]
times = pd.concat([pd.read_csv(f"output/{x}") for x in times])
times = times.drop(columns=["prompt"])
times = times.groupby(["scheduler", "num_inference_steps"]).mean().reset_index()

# Plot time wrt to inference steps, one curve per scheduler.
fig, ax = plt.subplots()
for scheduler in times["scheduler"].unique():
    ax.plot(times[times["scheduler"] == scheduler]["num_inference_steps"], times[times["scheduler"] == scheduler]["time"], label=scheduler)
print(times[times["scheduler"] == "linear"])
ax.set_xlabel("Number of inference steps")
ax.set_ylabel("Time (s)")
ax.legend()
fig.tight_layout()
fig.savefig("time.png")
