# Mini-Shakespearean LLM

![Loss Convergence Plot](my-transformer/loss_curve.png)

## Project Overview

This project builds a small GPT-style transformer language model trained on the Tiny Shakespeare dataset. The goal is to understand how transformer architecture works by implementing the major parts ourselves using PyTorch, including tokenization, embeddings, self-attention, MLP blocks, the language model head, training loop, and generation.

The project compares two local models:

- **Model A:** Baseline compact transformer
- **Model B:** Scaled transformer with increased model capacity

Both models are evaluated using training loss, validation loss, perplexity, and generated Shakespeare-style text.

## Repository Layout

```text
Mini-Shakespearean-LLM/
├── my-transformer/
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── generate.py
│   ├── plot_losses.py
│   ├── loss_curve.png
│   ├── logs/
│   │   ├── loss_A.csv
│   │   └── loss_B.csv
│   └── checkpoints/
│       ├── model_A.pt
│       └── model_B.pt
├── evaluation/
│   ├── benchmark_quantitative.py
│   ├── benchmark_qualitative.py
│   ├── quantitative_results.md
│   └── qualitative_comparison.md
├── requirements.txt
└── README.md
```

## Model Configurations

| Model | Type | Description |
|---|---|---|
| Model A | Baseline | Smaller GPT-style transformer used as the compact comparison model |
| Model B | Scaled | Larger GPT-style transformer with more capacity for comparison |

## Tokenization

This project uses a byte-level tokenizer. The raw Shakespeare text is encoded into UTF-8 byte integers.

```python
tokens = list(text.encode("utf-8"))
```

This gives a fixed vocabulary size of 256 tokens, which avoids string parsing problems and makes the input pipeline simple and stable.

## How to Install

Clone the repository and install the required packages:

```bash
git clone https://github.com/shashankreddygillella/Mini-Shakespearean-LLM-.git
cd Mini-Shakespearean-LLM-
pip install -r requirements.txt
```

## How to Train Model A

```bash
cd my-transformer
python train.py --config A --steps 3000
```

## How to Train Model B

```bash
cd my-transformer
python train.py --config B --steps 3000
```

## How to Generate Text

Generate text using Model A:

```bash
cd my-transformer
python generate.py --model A --prompt "To be, or not to "
```

Generate text using Model B:

```bash
cd my-transformer
python generate.py --model B --prompt "To be, or not to "
```

## How to Recreate the Loss Plot

```bash
cd my-transformer
python plot_losses.py
```

The generated plot is saved as:

```text
my-transformer/loss_curve.png
```

## Evaluation

The evaluation folder contains scripts and written outputs for both quantitative and qualitative evaluation.

Quantitative evaluation compares:

- Final training loss
- Final validation loss
- Perplexity

Qualitative evaluation compares generated completions from:

- Custom Model A
- Custom Model B
- Gemini Flash or another free-tier production LLM

## Summary

This project shows that a mini transformer can learn basic Shakespeare-style patterns, but the generated text is still limited compared to production LLMs. Model B is expected to perform better than Model A because it has more capacity, but both models are still small compared to industrial-scale systems.
