"""
benchmark_quantitative.py — Cross-entropy loss + Perplexity for Model A vs B
on a held-out evaluation slice.

Usage (run from evaluation/):
    python benchmark_quantitative.py
"""
import math
import sys
import torch

sys.path.append("../my-transformer")
from model import GPT
from tokenizer import encode, VOCAB_SIZE


def load_model(config_letter, device):
    ckpt = torch.load(f"../my-transformer/checkpoints/model_{config_letter}.pt", map_location=device)
    model = GPT(vocab_size=VOCAB_SIZE, **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, data, block_size, device, n_batches=50, batch_size=32):
    total_loss = 0.0
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / n_batches


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("../input.txt", "r") as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    held_out = data[int(0.9 * len(data)):]  # same val slice used during training

    print(f"{'Model':<10}{'CrossEntropyLoss':<20}{'Perplexity':<15}")
    for cfg in ["A", "B"]:
        model = load_model(cfg, device)
        loss = evaluate(model, held_out, model.config["block_size"], device)
        ppl = math.exp(loss)
        print(f"{cfg:<10}{loss:<20.4f}{ppl:<15.2f}")


if __name__ == "__main__":
    main()
