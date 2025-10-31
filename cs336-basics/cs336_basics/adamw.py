from collections.abc import Callable
from typing import Optional
import torch
import math

def gradient_clipping(parameters, max_l2_norm, eps=10**-6):
    grads = [p.grad for p in parameters if getattr(p, "grad", None) is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm >= max_l2_norm:
        for g in grads:
            g.data *= max_l2_norm / (total_norm + eps)
    

def learning_rate_schedule(t, lr_max, lr_min, tw, tc):
    if t < tw:
        return t / tw * lr_max    
    if t <= tc:
        return lr_min + 0.5 * (1 + math.cos((t - tw) / (tc - tw) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=10**-8, weight_decay=0.1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
                    "lr": lr,
                    "beta1": betas[0],
                    "beta2": betas[1],
                    "eps": eps,
                    "weight_decay": weight_decay
                }
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 1
                
                m = state["m"]
                v = state["v"]
                t = state["t"] # Get iteration number from the state, or initial value.
                
                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                state["m"] = m = beta1 * m + (1 - beta1) * grad
                state["v"] = v = beta2 * v + (1 - beta2) * grad ** 2
                
                lr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                p.data -= lr * m / (v ** 0.5 + eps) # Update weight tensor in-place.
                p.data -= group["lr"] * weight_decay * p.data # Apply weight decay
                
                state["t"] += 1
        
        return loss 
    

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = AdamW([weights], lr=1)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step()