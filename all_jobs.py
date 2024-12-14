from schedule_variation import generate_image
import itertools

prompts = [
    "A photo of llama wearing sunglasses standing on the deck of a spaceship with the Earth in the background.",
    "a shiba inu wearing a beret and black turtleneck",
    "a young girl playing piano",
    "A train ride in the monsoon rain in Kerala. With a Koala bear wearing a hat looking out of the window. There is a lot of coconut trees out of the window.",
]

jobs = list(itertools.product(
    prompts,
    ["linear", "exponential", "cosine"],
    [3, 4, 8, 15],
))

seeds = [42, 69, 223, 241, 242, 323, 365, 420, 431, 432]

if __name__ == "__main__":
    for seed in seeds:
        generate_image(jobs, seed)
