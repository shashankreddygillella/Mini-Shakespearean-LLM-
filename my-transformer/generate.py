"""Generate Shakespeare-style text using Model A or Model B."""

import argparse
from pathlib import Path

import torch
from model import GPTLanguageModel
from tokenizer import encode, decode, VOCAB_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B"], required=True)
    parser.add_argument("--prompt", type=str, default="To be, or not to ")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("checkpoints") / f"model_{args.model}.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = GPTLanguageModel(VOCAB_SIZE, **checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
