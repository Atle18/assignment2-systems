import torch

def softmax(x: torch.Tensor, dim: int):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    y = x_exp / x_exp.sum(dim=dim, keepdim=True)
    return y.to(in_dtype)