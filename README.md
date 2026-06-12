# Banking77 Intent Classification using DistilBERT

## Overview

This project compares a traditional machine learning approach (TF-IDF + Logistic Regression) with a transformer-based approach (DistilBERT) for customer intent classification on the Banking77 (HuggingFace) benchmark dataset.

The objective is to automatically classify customer support messages into one of 77 banking-related intent categories, enabling faster and more efficient customer service routing.

---

## Dataset

**Banking77** is a public NLP benchmark dataset consisting of online banking customer support queries.

* 77 intent classes
* 10,003 training samples
* 3,080 test samples
* English language customer support messages

Dataset source:

https://huggingface.co/datasets/PolyAI/banking77

---

## Models

### Baseline: TF-IDF + Logistic Regression

* TF-IDF vectorization
* English stopword removal
* Unigrams and bigrams
* Maximum 15,000 features
* Logistic Regression classifier

### Transformer Model: DistilBERT

* Model: distilbert-base-uncased
* Hugging Face Transformers
* Sequence length: 64
* Batch size: 8
* Weight decay: 0.01
* 10 training epochs
* Best checkpoint automatically selected

---

## Results

| Model                        | Accuracy | Macro F1 Score |
| ---------------------------- | -------- | -------------- |
| TF-IDF + Logistic Regression | 82.08%   | 81.99%         |
| DistilBERT                   | 93.15%   | 93.13%         |

DistilBERT significantly outperformed the traditional machine learning baseline, demonstrating the effectiveness of transformer-based architectures for intent classification tasks.

---

## Example Predictions

| Input                            | Prediction               |
| -------------------------------- | ------------------------ |
| my card has not arrived yet      | card_arrival             |
| cash withdrawal failed           | declined_cash_withdrawal |
| I forgot my pin code             | pin_blocked              |
| bank transfer did not go through | failed_transfer          |

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-Learn
* Pandas
* NumPy
* NLP
* Deep Learning
* Text Classification

---

## Repository Structure

```text
.
├── banking77_baseline.py
├── Project_Report.pdf
├── requirements.txt
└── README.md
```

---

## Future Work

* Hyperparameter optimization
* Evaluation of larger transformer architectures
* Data augmentation techniques
* Deployment as a real-time inference API

---

## Academic Note

This project was developed as part of the **VBM688 – Fundamentals of Artificial Intelligence** course within the M.Sc. Data and Knowledge Engineering program at Hacettepe University.

The repository contains the implementation, experiments, model evaluation, and project report prepared for the course project. The study focuses on comparing traditional machine learning methods with transformer-based architectures for banking customer intent classification using the Banking77 benchmark dataset.
