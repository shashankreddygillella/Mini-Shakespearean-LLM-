"""
train.py — train Model A (baseline) or Model B (scaled) on tinyshakespeare.

Usage:
    python train.py --config A --steps 2000
    python train.py --config B --steps 2000

Writes:
    checkpoints/model_{config}.pt
    logs/loss_{config}.csv   (step, train_loss, val_loss)
"""
import argparse
import csv
import os
import torch

from model import GPT, count_params
from tokenizer import encode, VOCAB_SIZE

CONFIGS = {
    # Model A: shallow baseline
    "A": dict(n_layer=2, n_head=4, n_embd=128, block_size=64),
    # Model B: scaled up — more depth, wider embeddings, longer context
    "B": dict(n_layer=4, n_head=8, n_embd=256, block_size=128),
}


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters=20):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B"], required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data", type=str, default="../input.txt")
    parser.add_argument("--eval_every", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    cfg = CONFIGS[args.config]
    print(f"Training Model {args.config}: {cfg}")

    with open(args.data, "r") as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = GPT(vocab_size=VOCAB_SIZE, **cfg).to(device)
    print(f"Parameter count: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/loss_{args.config}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])

    for step in range(args.steps):
        xb, yb = get_batch(train_data, cfg["block_size"], args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            val_loss = estimate_loss(model, val_data, cfg["block_size"], args.batch_size, device)
            print(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f}")
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, loss.item(), val_loss])

    torch.save({"model_state": model.state_dict(), "config": cfg}, f"checkpoints/model_{args.config}.pt")
    print(f"Saved checkpoints/model_{args.config}.pt")


if __name__ == "__main__":
    main()
