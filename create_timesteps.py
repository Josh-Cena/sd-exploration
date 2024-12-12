from typing import Literal
import numpy as np

def create_timesteps(scheduler: Literal["linear", "cosine", "exponential"], num_inference_steps: int) -> np.ndarray:
    if scheduler == "linear":
        step_range = np.linspace(999, 0, num_inference_steps)
    elif scheduler == "cosine":
        step_range = np.cos(np.linspace(0, np.pi, num_inference_steps)) * 499 + 500
    elif scheduler == "exponential":
        step_range = 1000 - np.exp(np.linspace(0, np.log(50), num_inference_steps)) * 20
    return np.clip(step_range, 0, 999).astype(int)
