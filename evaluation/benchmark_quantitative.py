"""Quantitative benchmark for Mini-Shakespearean LLM models."""

import math

# Replace these example values with values from your final training logs if needed.
RESULTS = {
    "Model A": {"final_cross_entropy_loss": 2.55},
    "Model B": {"final_cross_entropy_loss": 2.30},
}

for model_name, metrics in RESULTS.items():
    loss = metrics["final_cross_entropy_loss"]
    perplexity = math.exp(loss)
    print(f"{model_name}: final loss={loss:.4f}, perplexity={perplexity:.4f}")
