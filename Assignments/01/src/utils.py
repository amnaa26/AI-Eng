import random
import numpy as np
import torch
import yaml


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
