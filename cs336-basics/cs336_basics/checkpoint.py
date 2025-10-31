import torch

def save_checkpoint(model, optimizer, iteration, out):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(ckpt, out)

def load_checkpoint(src, model, optimizer) -> int:
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    iteration = ckpt["iteration"]
    return iteration
