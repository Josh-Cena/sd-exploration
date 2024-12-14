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

if __name__ == "__main__":
    generate_image(jobs, 42)
    generate_image(jobs, 69)
    generate_image(jobs, 223)
    generate_image(jobs, 241)
    generate_image(jobs, 242)
    generate_image(jobs, 323)
    generate_image(jobs, 365)
    generate_image(jobs, 420)
    generate_image(jobs, 431)
    generate_image(jobs, 432)
