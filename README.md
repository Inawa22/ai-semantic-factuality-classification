# Semantic Factuality Classification

As part of the Datacamp Data4Good Challenge, this project checks whether an AI-generated educational answer is:

* **Factual:** correct
* **Contradictory:** incorrect
* **Irrelevant:** unrelated to the question

The aim is to make AI learning tools more trustworthy by spotting misleading or off-topic answers.

## How It Works

The model uses:

* TF-IDF to understand important words and phrases
* Sentence embeddings to compare the question, context and answer
* Class-balanced Logistic Regression for classification

Combining lexical and semantic features helped the model detect contradictions and irrelevant answers better than TF-IDF alone.

## Results

**Validation Macro F1: 0.81**

| Class         | F1 score |
| ------------- | -------: |
| Factual       |     0.96 |
| Contradiction |     0.55 |
| Irrelevant    |     0.91 |

## Data

The training dataset contains **21,021 examples**, while the test dataset contains **2,000 examples**.

Each example includes:

* A question
* Supporting context
* An AI-generated answer
* A classification label

## Run the Project

```bash
pip install -r requirements.txt
python src/train_and_predict.py
```

## Project Structure

* `data/` — training and test data
* `notebooks/` — experiments and analysis
* `src/` — reusable code and training pipeline
* `submission.csv` — example predictions

Built for the **4th Annual Data4Good Factuality Challenge**.
