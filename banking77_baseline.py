from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import evaluate

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# LOAD DATASET
# ==========================================================

dataset = load_dataset("PolyAI/banking77")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

label_names = dataset["train"].features["label"].names

train_df["label_name"] = train_df["label"].apply(lambda x: label_names[x])
test_df["label_name"] = test_df["label"].apply(lambda x: label_names[x])

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Number of labels:", len(label_names))

# ==========================================================
# BASELINE MODEL
# ==========================================================

baseline_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=15000,
            ngram_range=(1, 2)
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            random_state=42,
            n_jobs=-1
        )
    )
])

baseline_model.fit(train_df["text"], train_df["label"])

baseline_preds = baseline_model.predict(test_df["text"])

baseline_acc = accuracy_score(
    test_df["label"],
    baseline_preds
)

baseline_f1 = f1_score(
    test_df["label"],
    baseline_preds,
    average="macro"
)

print("\n=== BASELINE RESULTS ===")
print("Accuracy:", baseline_acc)
print("Macro F1:", baseline_f1)

# ==========================================================
# DISTILBERT
# ==========================================================

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

tokenized_dataset = tokenized_dataset.rename_column(
    "label",
    "labels"
)

tokenized_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_names)
)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    macro_f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "macro_f1": macro_f1["f1"]
    }

# ==========================================================
# TRAINING SETTINGS
# ==========================================================

training_args = TrainingArguments(
    output_dir="./banking77_distilbert",

    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",

    num_train_epochs=10,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,

    save_total_limit=2,

    report_to="none"
)

# ==========================================================
# TRAINER
# ==========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ==========================================================
# TRAIN
# ==========================================================

print("\n=== DISTILBERT TRAINING STARTED ===")

trainer.train()

# ==========================================================
# BEST MODEL EVALUATION
# ==========================================================

bert_results = trainer.evaluate()

print("\n=== BEST DISTILBERT RESULTS ===")
print(bert_results)

print("\nBest checkpoint:")
print(trainer.state.best_model_checkpoint)

print("\nBest metric:")
print(trainer.state.best_metric)

# ==========================================================
# COMPARISON
# ==========================================================

comparison_df = pd.DataFrame([
    {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": baseline_acc,
        "Macro F1": baseline_f1
    },
    {
        "Model": "DistilBERT (Best Epoch)",
        "Accuracy": bert_results["eval_accuracy"],
        "Macro F1": bert_results["eval_macro_f1"]
    }
])

print("\n=== MODEL COMPARISON ===")
print(comparison_df)

# ==========================================================
# SAMPLE PREDICTIONS
# ==========================================================

id2label = {
    i: label
    for i, label in enumerate(label_names)
}

def predict_intent_bert(
    text,
    model,
    tokenizer,
    id2label
):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    return id2label[pred_id]

example_texts = [
    "my card has not arrived yet",
    "cash withdrawal failed",
    "I forgot my pin code",
    "bank transfer did not go through"
]

print("\n=== SAMPLE PREDICTIONS ===")

for text in example_texts:

    prediction = predict_intent_bert(
        text,
        model,
        tokenizer,
        id2label
    )

    print(f"Text: {text}")
    print(f"Prediction: {prediction}")
    print("-" * 50)

# ==========================================================
# SAVE MODEL
# ==========================================================

save_path = "./saved_banking77_distilbert"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("\nModel saved successfully.")
print("Saved to:", save_path)