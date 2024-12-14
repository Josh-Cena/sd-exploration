import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from typing import cast

def extract_images_from_npz(npz_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(npz_folder):
        if not file.endswith(".npz"):
            continue
        npz_path = os.path.join(npz_folder, file)
        print(f"Processing {npz_path}...")

        data = np.load(npz_path)
        images = cast(np.ndarray, data["data"])
        # The image itself is in CWH format, so we need to transpose it to WHC
        images = images.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)

        # Save each image
        for i, img_array in tqdm(list(enumerate(images))):
            img = Image.fromarray(img_array, mode="RGB")
            img.save(os.path.join(output_folder, f"{file[:-4]}_{i}.png"))


if __name__ == "__main__":
    npz_folder = "Imagenet64_train_part1_npz/"
    output_folder = "real_images/"
    extract_images_from_npz(npz_folder, output_folder)
