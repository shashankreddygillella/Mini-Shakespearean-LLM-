# Financial RAG Challenge - OfficeQA

## Project Overview

This project builds a Retrieval-Augmented Generation (RAG) system using Databricks OfficeQA Treasury Bulletin data. The project compares a simple Baseline RAG system with an Engineered RAG system to understand how chunking, metadata, and retrieval filtering affect performance.

## Dataset

- Data Source: Databricks OfficeQA
- Format Used: TXT / Markdown Treasury Bulletin files
- Years Used: 2022, 2023, 2024, 2025
- Answer Key: officeqa_full.csv
- Evaluation Cutoff: K = 5

## Technical Stack

- Vector Database: FAISS
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Metadata: Year and Month tags stored for each chunk
- Baseline Chunking: 1200-character chunks with 100-character overlap
- Engineered Chunking: 1800-character section-aware chunks with 250-character overlap
- Evaluation Metrics: Hit Rate@5, MRR, Groundedness, Factual Accuracy, and Hallucination Rate

## Repository Structure

```text
financial-rag-officeqa/
├── data/
│   └── docs/
├── results/
│   ├── baseline_results.csv
│   ├── engineered_results.csv
│   └── scorecard.csv
├── financial_rag_officeqa_colab.ipynb
├── README.md
└── requirements.txt
```

## Architecture

```text
OfficeQA TXT / Markdown Files
        ↓
Load Treasury Bulletin Documents
        ↓
Create Baseline and Engineered Chunks
        ↓
Generate Embeddings using MiniLM
        ↓
Store and Search Vectors using FAISS
        ↓
Retrieve Top 5 Chunks
        ↓
Generate Answer from Retrieved Context
        ↓
Compare with officeqa_full.csv Answer Key
        ↓
Calculate Baseline vs Engineered Metrics
```

## Scorecard

| Metric | Baseline (Simple) | Engineered (Improved) |
|---|---:|---:|
| Hit Rate@5 | 33.3% | 22.2% |
| MRR | 0.19 | 0.06 |
| Groundedness | 100.0% | 100.0% |
| Factual Accuracy | 0.0% | 0.0% |
| Hallucination Rate | 0.0% | 0.0% |

## Engineering Reflection

### 1. The Bottleneck

The main bottleneck was retrieval. The baseline Hit Rate@5 was 33.3% and the MRR was 0.19. This means the correct evidence was not always found in the top five chunks, and even when it was found, it was not always ranked near the top.

Groundedness was 100%, but factual accuracy was 0%. This shows that the answer generator stayed within the retrieved context, but the retrieved context did not always contain the exact information needed to match the answer key.

### 2. The Metadata Fix

The engineered version used Year and Month metadata filtering with section-aware chunking. However, in this experiment, the engineered version did not improve the scores. Hit Rate@5 decreased from 33.3% to 22.2%, and MRR decreased from 0.19 to 0.06.

This likely happened because the metadata filter narrowed the search space too much or because some Year/Month tags were imperfect. This was an important lesson because metadata is powerful, but it must be accurate and tested carefully.

### 3. Scaling Insight

If this system scaled from four recent years to the full 1939–2025 archive, the first bottleneck would likely be indexing and retrieval speed. The number of chunks would increase significantly, which would increase embedding generation time, vector database size, and search cost.

To scale the system, I would use stronger metadata partitioning, batch embedding generation, persistent FAISS indexes, and possibly a reranking step. I would also improve document parsing and table-aware chunking so that financial tables are not split incorrectly.

## Key Learning

This project showed that the retriever is one of the most important parts of a RAG system. The generator can only answer correctly if the retrieved context contains the correct evidence. The baseline was useful because it gave a comparison point. In this run, the engineered system did not outperform the baseline, but that result is still useful because it shows that engineering changes must be proven with real metrics instead of assumed to be better.

## Results Files

- `results/baseline_results.csv`
- `results/engineered_results.csv`
- `results/scorecard.csv`

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open and run:

```text
financial_rag_officeqa_colab.ipynb
```

The notebook loads the OfficeQA data, builds baseline and engineered RAG systems, evaluates both systems, and saves the final results into the `results/` folder.
