import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_similarities(file_name: str):
    similarities = [f"{x}/similarities.csv" for x in os.listdir("output")]
    similarities = pd.concat([pd.read_csv(f"output/{x}") for x in similarities])
    similarities = similarities.drop(columns=["prompt"])
    similarities = similarities.groupby(["scheduler", "num_inference_steps"]).mean().reset_index()

    fig, ax = plt.subplots()
    for scheduler in similarities["scheduler"].unique():
        ax.plot(similarities[similarities["scheduler"] == scheduler]["num_inference_steps"], similarities[similarities["scheduler"] == scheduler]["similarity"], label=scheduler)
    ax.set_xlabel("Number of inference steps")
    ax.set_ylabel("CLIP score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(file_name)

if __name__ == "__main__":
    plot_similarities("plots/similarity.png")
