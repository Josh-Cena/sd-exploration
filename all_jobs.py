from schedule_variation import generate_image
import itertools

jobs = itertools.product(
    [
        "A photo of llama wearing sunglasses standing on the deck of a spaceship with the Earth in the background.",
        "a shiba inu wearing a beret and black turtleneck",
        "a young girl playing piano",
        "A train ride in the monsoon rain in Kerala. With a Koala bear wearing a hat looking out of the window. There is a lot of coconut trees out of the window.",
    ],
    ["linear", "exponential", "cosine"],
    [3, 4, 8, 15],
)

generate_image(jobs, 431)
