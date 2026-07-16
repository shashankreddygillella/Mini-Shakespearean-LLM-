"""
benchmark_qualitative.py — Generate completions from Model A, Model B, and a
free-tier production LLM (Gemini Flash) for the same prompts, and dump a
markdown comparison table you can paste into your report.

Gemini call requires your own API key:
    export GEMINI_API_KEY=your_key_here
    pip install google-generativeai --break-system-packages

Usage:
    python benchmark_qualitative.py
"""
import os
import sys
import torch

sys.path.append("../my-transformer")
from model import GPT
from tokenizer import encode, decode, VOCAB_SIZE

PROMPTS = [
    "To be, or not to ",
    "O Romeo, Romeo, wherefore ",
    "Now is the winter of our ",
    "Friends, Romans, countrymen, ",
]

NUM_TOKENS = 150


def load_model(config_letter, device):
    ckpt = torch.load(f"../my-transformer/checkpoints/model_{config_letter}.pt", map_location=device)
    model = GPT(vocab_size=VOCAB_SIZE, **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def generate_local(model, prompt, device):
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=NUM_TOKENS, temperature=0.8)
    return decode(out[0].tolist())


def generate_gemini(prompt):
    """Calls Gemini Flash if GEMINI_API_KEY is set; otherwise returns a
    placeholder so the script still runs end to end."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[Set GEMINI_API_KEY to fetch a real completion here]"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(
            f"Continue this text in the style of Shakespeare, "
            f"about {NUM_TOKENS // 4} words: {prompt}"
        )
        return resp.text
    except Exception as e:
        return f"[Gemini call failed: {e}]"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_a = load_model("A", device)
    model_b = load_model("B", device)

    rows = []
    for prompt in PROMPTS:
        out_a = generate_local(model_a, prompt, device)
        out_b = generate_local(model_b, prompt, device)
        out_gemini = generate_gemini(prompt)
        rows.append((prompt, out_a, out_b, out_gemini))

    print("| Prompt | Model A | Model B | Gemini Flash |")
    print("|---|---|---|---|")
    for prompt, a, b, g in rows:
        clean = lambda s: s.replace("\n", " ").replace("|", "/")
        print(f"| {clean(prompt)} | {clean(a)} | {clean(b)} | {clean(g)} |")


if __name__ == "__main__":
    main()
