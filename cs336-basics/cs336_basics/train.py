"""
python train.py \
    --data-path data/tokens.npy \
    --vocab-size 10000 \
    --context-length 128 \
    --batch-size 8 \
    --d-model 64 --num-heads 4 --d-ff 128 --num-layers 3 \
    --lr 1e-3 --weight-decay 0.1 --max-iters 1000 \
    --save-path checkpoints/ckpt.pt

This script expects a 1D numpy array of token ids stored as a file (e.g. saved with
`np.save("tokens.npy", tokens_array)`), and will open it with `np.memmap` for memory-efficient
loading.
"""

import argparse
import os
import time
import numpy as np
import torch

from cs336_basics.llm import LLM
from cs336_basics.adamw import AdamW, gradient_clipping
from cs336_basics.data_loading import data_loading
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.checkpoint import save_checkpoint


def build_model(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta=None, device="cpu"):
    model = LLM(vocab_size=vocab_size, context_length=context_length, num_layers=num_layers,
                d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=theta)
    return model.to(device)


def get_optimizer(model, lr, weight_decay):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def load_data_memmap(path):
    # Open as read-only memmap; dtype int64 expected
    return np.load(path, mmap_mode="r")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None, help="Path to 1D numpy token ids file (.npy or raw memmap)")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-iters", type=int, default=1000)
    p.add_argument("--save-path", type=str, default="checkpoint.pt")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--theta", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load data memmap or build dummy data
    if args.data_path is not None and os.path.exists(args.data_path):
        data = load_data_memmap(args.data_path)
        print(f"Loaded data memmap from {args.data_path}, length={len(data)}")
    else:
        print("No data path provided or file not found. Creating small random dataset for demo.")
        data = np.random.randint(0, args.vocab_size, size=10000, dtype=np.int64)

    model = build_model(args.vocab_size, args.context_length, args.num_layers,
                        args.d_model, args.num_heads, args.d_ff, theta=args.theta,
                        device=device)

    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    iter = 0
    start_time = time.time()
    while iter < args.max_iters:
        model.train()
        inputs, targets = data_loading(data, args.batch_size, args.context_length, device)

        logits = model(inputs)  # shape: (B, T, V)
        # Flatten (B*T, V) and (B*T)
        B, T, V = logits.shape
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)

        loss = cross_entropy(logits_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()

        # clip gradients
        gradient_clipping(model.parameters(), max_l2_norm=1.0)

        optimizer.step()

        if iter % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"iter={iter} loss={loss.item():.6f} time={elapsed:.2f}s device={device}")

        if iter % args.save_interval == 0 and iter > 0:
            save_checkpoint(model, optimizer, iter, args.save_path + f".iter{iter}.pt")
            print(f"Saved checkpoint to {args.save_path}.iter{iter}.pt")

        iter += 1

    # final save
    save_checkpoint(model, optimizer, iter, args.save_path)
    print(f"Training finished. Final checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
