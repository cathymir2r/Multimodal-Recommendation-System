import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_best_epoch(epochs: list[dict], metric_key: str = "mse") -> dict:
    if not epochs:
        return {}
    valid_epochs = [epoch for epoch in epochs if epoch.get(metric_key) is not None]
    if not valid_epochs:
        return epochs[-1]
    return min(valid_epochs, key=lambda epoch: epoch[metric_key])
