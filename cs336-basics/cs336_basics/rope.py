import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        half = d_k // 2
        
        k = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2 * k / d_k)
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        ang = pos[:, None] * inv_freq[None, :]
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: [..., seq_len, d_k], token_positions: [..., seq_len]
        cos = self.cos[token_positions.to(x.device)]
        sin = self.sin[token_positions.to(x.device)]
        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x = x.view(*x.shape[:-1], self.d_k // 2, 2)
        x_even, x_odd = x[..., 0], x[..., 1]
        
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        y = torch.stack((y_even, y_odd), dim=-1).reshape(*x.shape[:-2], self.d_k)
        return y.to(in_dtype)
    
    