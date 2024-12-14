"""
FID scores obtained using the following command:
echo "scheduler,steps,FID" > fid.csv
for dir in output-alt/*; do
  fid_output=$(pytorch-fid real_images/ "$dir" --batch-size 64 --device cuda | grep 'FID: ' | sed 's/FID: //')
  scheduler=$(echo "$dir" | cut -d'_' -f1)
  steps=$(echo "$dir" | cut -d'_' -f2)
  echo "$scheduler,$steps,$fid_output" >> fid.csv
done
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fid.csv")
fig, ax = plt.subplots()
for scheduler, steps in df.groupby("scheduler"):
    ax.plot(steps["steps"], steps["FID"], label=scheduler)
ax.legend()
ax.set_xlabel("Steps")
ax.set_ylabel("FID")
fig.tight_layout()
fig.savefig("fid_score.png")
