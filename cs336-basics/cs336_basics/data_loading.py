import numpy as np
import torch

def data_loading(x, batch_size, context_length, device):
    """
    x: 1D numpy array (or np.memmap) of token ids
    returns: (inputs, targets) both LongTensor of shape (B, T) on `device`
    """
    n = int(x.shape[0])

    # sample start positions so that i+context_length is valid
    starts = np.random.randint(0, n - context_length, size=batch_size)

    offsets = np.arange(context_length)[None, :]          # (1, T)
    idx_inp = starts[:, None] + offsets                   # (B, T)

    inp = x[idx_inp].astype(np.int64, copy=False)         # (B, T)
    tgt = x[idx_inp + 1].astype(np.int64, copy=False)         # (B, T)

    inputs = torch.as_tensor(inp, dtype=torch.long, device=device)
    targets = torch.as_tensor(tgt, dtype=torch.long, device=device)
    return inputs, targets