import torch
import math

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.W1 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
        )
        self.W2 = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype)
        )
        self.W3 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
        )
        
        std = math.sqrt(2.0 / (self.d_ff + self.d_model))
        torch.nn.init.trunc_normal_(self.W1, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.W2, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.W3, mean=0.0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor):
        x1 = x @ self.W1.transpose(-1, -2)
        x3 = x @ self.W3.transpose(-1, -2)
        x1 = x1 * torch.sigmoid(x1)
        x2 = (x1 * x3) @ self.W2.transpose(-1, -2)
        return x2
