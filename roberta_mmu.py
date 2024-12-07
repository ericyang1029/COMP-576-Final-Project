import argparse
import numpy as np
import os
import gc
from pprint import pprint

import datasets
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
import torch
from sklearn.metrics import f1_score

gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Define label maps
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large", trust_remote_code=True, truncation=True, padding=True)

# def map_labels(example):
#     # Map string labels to integers
#     example['labels'] = label2id[example['label']]
#     return example

def tokenized_f(d):
    tokenized_inputs = tokenizer(
        d['text'],
        padding=True,
        truncation=True,
        max_length=128,
    )
    return tokenized_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, predictions, average='macro')

    num_labels = len(label2id)
    per_class_acc = {}
    for label_id in range(num_labels):
        label_indices = (labels == label_id)
        correct = (predictions[label_indices] == label_id).sum()
        total = label_indices.sum()
        acc = correct / total if total > 0 else 0.0
        per_class_acc[f'class_{label_id}_accuracy'] = acc

    results = {
        'f1_macro': f1_macro,
    }
    results.update(per_class_acc)
    return results

def sample_equal(dataset, label_col='label', n_per_label=50000, seed=42):
    unique_labels = set(dataset[label_col])
    sampled_splits = []
    for lab in unique_labels:
        subset = dataset.filter(lambda x: x[label_col] == lab)
        if len(subset) > n_per_label:
            subset = subset.select(range(n_per_label))
        sampled_splits.append(subset)
    balanced_dataset = datasets.concatenate_datasets(sampled_splits)
    balanced_dataset = balanced_dataset.shuffle(seed=seed)
    return balanced_dataset

def load_and_process_data():
    # Load dataset
    dataset = datasets.load_dataset("Brand24/mms")

    # Filter only English data in the train set
    dataset['train'] = dataset['train'].filter(lambda x: x['language'] == 'en')

    # Map labels
    # dataset['train'] = dataset['train'].map(map_labels)

    # Sample 50,000 examples per label from the training set
    dataset['train'] = sample_equal(dataset['train'], label_col='label', n_per_label=50000)

    # Keep only relevant columns
    drop_cols_train = [col for col in dataset['train'].column_names if col not in ['text', 'label']]
    dataset['train'] = dataset['train'].remove_columns(drop_cols_train)

    # Split off 1% as validation
    split_dataset = dataset['train'].train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenized_f, batched=True)
    val_dataset = val_dataset.map(tokenized_f, batched=True)

    return train_dataset, val_dataset

if __name__ == "__main__":
    num_of_labels = 3

    parser = argparse.ArgumentParser(description="Fine-tune on Brand24/mms English subset with validation split")
    parser.add_argument('--model', type=str, help="Model ID to load for finetuning. E.g. 'roberta'")
    args, unknown = parser.parse_known_args()

    # Load and process data
    train_dataset, val_dataset = load_and_process_data()
    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))

    # Load model
    if args.model == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-large",
            num_labels=num_of_labels,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
        )
    else:
        print("Please specify a supported model (e.g. roberta).")
        exit(1)

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        output_dir='./results',
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        seed=42,
        save_steps=500,
        weight_decay=0.01,
        warmup_steps=200,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Evaluate on the validation set
    results = trainer.evaluate()
    print("Validation Results:", results)

    # Save tokenizer and model
    tokenizer.save_pretrained(f"{args.model}-final-tokenizer")
    trainer.save_model(f"{args.model}-final-finetuned")
    print("Model saved")