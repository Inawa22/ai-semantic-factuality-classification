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

##Context

Built as part of the Data4Good Factuality Challenge.

# Data4Good Case Challenge



![Data4Good](Data4Good.png)



## Background
Artificial Intelligence (AI) is rapidly transforming education by providing students with instant access to information and adaptive learning tools. Still, it also introduces significant risks, such as the spread of misinformation and fabricated content. Research indicates that large language models (LLMs) often confidently generate factually incorrect or “hallucinated” responses, which can mislead learners and erode trust in digital learning platforms. 

The 4th Annual Data4Good Competition challenges participants to develop innovative analytics solutions to detect and improve factuality in AI-generated educational content, ensuring that AI advances knowledge rather than confusion.

## The data

The data provided is a Questions/Answer dataset to determine if the answer is factual, not factual (contradiction), or irrelevant to the question.


- Question: The question asked/prompted for
- Context: Relevant contextual support for the question
- Answer: The answer provided by an AI
- Type:  A categorical variable with three possible levels – Factual, Contradiction, Irrelevant:
  - Factual: the answer is correct
  - Contradiction: the answer is incorrect
  - Irrelevant: the answer has nothing to do with the question
  
There are 21,021 examples in the dataset (`data/train.json`) that you will experiment with. 


The test dataset (`data/test.json`) contains 2000 examples that you predict as one of the three provided classes. In addition to classification performance we are seeking as detailed as possible methodologies of your step-by-step approach in your notebooks. Discuss what worked well, what did not work well, and your suggestions or ideas if a general approach to these types of problems might exist.


