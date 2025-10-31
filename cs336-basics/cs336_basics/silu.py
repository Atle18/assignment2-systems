import torch

def silu(x):
    return x / (1.0 + torch.exp(-x))