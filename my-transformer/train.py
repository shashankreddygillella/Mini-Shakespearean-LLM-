"""Train Model A or Model B on Tiny Shakespeare."""

import argparse
import csv
from pathlib import Path

import torch
from model import GPTLanguageModel
from tokenizer import encode, VOCAB_SIZE


CONFIGS = {
    "A": {"block_size": 64, "n_embd": 128, "n_head": 4, "n_layer": 2, "dropout": 0.2},
    "B": {"block_size": 128, "n_embd": 256, "n_head": 4, "n_layer": 4, "dropout": 0.2},
}


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=20):
    model.eval()
    result = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        result[split] = losses.mean().item()
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B"], required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--input", type=str, default="../input.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CONFIGS[args.config]
    text = Path(args.input).read_text(encoding="utf-8")
    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = GPTLanguageModel(VOCAB_SIZE, **cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    log_path = Path("logs") / f"loss_{args.config}.csv"

    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])

        for step in range(args.steps):
            if step % 100 == 0 or step == args.steps - 1:
                losses = estimate_loss(model, train_data, val_data, cfg["block_size"], args.batch_size, device)
                writer.writerow([step, losses["train"], losses["val"]])
                print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = get_batch(train_data, cfg["block_size"], args.batch_size, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, Path("checkpoints") / f"model_{args.config}.pt")
    print(f"Saved Model {args.config}")


if __name__ == "__main__":
    main()
