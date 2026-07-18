# Mini-Shakespearean LLM

![Model A vs Model B Loss Curve](my-transformer/loss_curve.png)

## Project Overview

This project builds a small GPT-style transformer language model trained on the Tiny Shakespeare dataset. The goal is to understand how a transformer language model works at a practical level: tokenization, embeddings, self-attention, feed-forward layers, next-token prediction, training loss, validation loss, perplexity, and text generation.

The project compares two local models:

- **Model A:** a smaller baseline transformer
- **Model B:** a scaled transformer with increased model capacity

Both models are trained locally and evaluated using quantitative metrics and qualitative text generation prompts.

## Repository Structure

```text
Mini-Shakespearean-LLM-/
├── input.txt
├── README.md
├── requirements.txt
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
└── docs/
    ├── reflection_report.md
    ├── roles_and_contributions.md
    ├── architecture_mapping_instructions.md
    └── submission_checklist.md
```

## Model Configurations

| Model | Purpose | Description |
|---|---|---|
| Model A | Baseline | Smaller model used as the compact reference point |
| Model B | Scaled | Larger model used to test whether increased model capacity improves learning |

The exact configuration is controlled inside the training code. Model A and Model B are trained separately and saved as checkpoints.

## Tokenization

This project uses a simple byte-level tokenizer. The text is converted into UTF-8 byte integers, giving a fixed vocabulary size of **256 tokens**. This avoids character parsing problems and allows the model to process raw Shakespeare text as a continuous stream.

## Training and Evaluation Summary

The two models were trained for 3000 steps and evaluated using training loss, validation loss, and perplexity.

| Model | Final Train Loss | Final Validation Loss | Validation Perplexity |
|---|---:|---:|---:|
| Model A | 1.6951 | 1.8262 | 6.21 |
| Model B | 1.3448 | 1.5948 | 4.93 |

Model B performs better than Model A because it reaches a lower validation loss and lower perplexity. This shows that the scaled model learned the Shakespeare text patterns more effectively than the smaller baseline model.

## How to Install

Clone the repository and install the requirements:

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

```bash
cd my-transformer
python generate.py --config A --prompt "To be, or not to " --max_new_tokens 150
python generate.py --config B --prompt "To be, or not to " --max_new_tokens 150
```

## How to Plot Loss Curves

```bash
cd my-transformer
python plot_losses.py
```

The output plot is saved as:

```text
my-transformer/loss_curve.png
```

## How to Run Quantitative Evaluation

```bash
python evaluation/benchmark_quantitative.py
```

This evaluation compares Model A and Model B using cross-entropy loss and perplexity.

## How to Run Qualitative Evaluation

```bash
python evaluation/benchmark_qualitative.py
```

This evaluation compares generated text from:

- Model A
- Model B
- Gemini Flash

The comparison focuses on Shakespearean style, structural stability, and repetition loops.

## Key Learning

The main learning from this project is that a language model is not magic. It learns by repeatedly predicting the next token, measuring the error, and updating its weights. A larger model can often learn better patterns, but it also requires more compute and careful evaluation. The local mini-models do not generate perfect Shakespeare, but they show the real mechanics behind modern LLMs at a small scale.

## Team Members

See `docs/roles_and_contributions.md` for the team role breakdown.

## Notes for Submission

The handwritten architecture mapping diagram should be uploaded separately as an image or PDF. Do not submit a fully typed architecture diagram if the assignment asks for hand annotation.
