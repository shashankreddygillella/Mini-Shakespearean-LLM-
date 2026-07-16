"""
generate.py — load a trained checkpoint and generate text from a prompt.

Usage:
    python generate.py --config A --prompt "To be, or not to " --tokens 150
"""
import argparse
import torch

from model import GPT
from tokenizer import encode, decode, VOCAB_SIZE


def load_model(config_letter, device):
    ckpt = torch.load(f"checkpoints/model_{config_letter}.pt", map_location=device)
    model = GPT(vocab_size=VOCAB_SIZE, **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B"], required=True)
    parser.add_argument("--prompt", type=str, default="To be, or not to ")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.config, device)

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
