# ai-semantic-factuality-classification
Semantic classification of AI-generated educational answers to detect factuality, contradiction, and irrelevance.

# Semantic Factuality Classification

This project classifies AI-generated educational answers into three categories:
- factual
- contradiction
- irrelevant

The goal is to improve trust in AI-assisted education by detecting misinformation and misalignment.

## Approach

We combine:
- TF-IDF with unigrams and bigrams
- Semantic similarity between question, context, and answer using sentence embeddings
- A class-balanced Logistic Regression classifier

This hybrid approach significantly improves detection of irrelevance and contradiction compared to lexical baselines alone.

## Results

Validation Macro F1: **0.81**

| Class | F1 |
|------|-----|
| factual | 0.96 |
| contradiction | 0.55 |
| irrelevant | 0.91 |

## Structure

- data/ — datasets (not included if restricted)
- notebooks/ — exploration and experiments
- src/ — reusable utilities and pipeline code
- submission.csv — example output

## How to run

```bash
pip install -r requirements.txt
python src/train_and_predict.py
