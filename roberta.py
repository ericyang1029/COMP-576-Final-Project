import pandas as pd
import shutil, argparse
import numpy as np
import os
from pprint import pprint
import datasets
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq,
BitsAndBytesConfig,DataCollatorForLanguageModeling, RobertaTokenizer, RobertaForSequenceClassification, AlbertTokenizer,
AlbertForSequenceClassification, EarlyStoppingCallback)

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gc
from sklearn.metrics import f1_score

gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Define label maps
label2id = {"neg": 0, "neutral": 1, "pos": 2}
id2label = {v: k for k, v in label2id.items()}

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large", trust_remote_code=True, truncation=True, padding=True)
model_id = "roberta-final-finetuned"
def map_labels(example):
    # Map string labels to integers
    example['labels'] = label2id[example['sentimentLabel']]
    return example

def tokenized_f(d):
    # Tokenize the comment only, as it's a single sentence classification
    tokenized_inputs = tokenizer(
        d['comment'],
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    return tokenized_inputs

def load_and_process_data(filename):
    # Load from CSV
    dataset_loaded = datasets.load_dataset("csv", data_files=filename)

    dataset_loaded['train'] = dataset_loaded['train'].map(map_labels)

    # Keep only comment and labels
    drop_cols = [col for col in dataset_loaded['train'].column_names 
                 if col not in ['comment', 'labels']]
    dataset_loaded['train'] = dataset_loaded['train'].remove_columns(drop_cols)

    # Tokenize dataset
    tokenized_ds = dataset_loaded.map(
        tokenized_f,
        batched=True
    )

    # Split data: 80% train, 20% validation
    split_ds = tokenized_ds['train'].train_test_split(test_size=0.2, shuffle=True, seed=42)

    return split_ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Macro F1
    f1_macro = f1_score(labels, predictions, average='macro')

    # Per-class accuracy
    num_labels = len(label2id)
    per_class_acc = {}
    for label_id in range(num_labels):
        # Indices where the true label is label_id
        label_indices = (labels == label_id)
        # Among these, how many did we predict correctly?
        correct = (predictions[label_indices] == label_id).sum()
        total = label_indices.sum()
        acc = correct / total if total > 0 else 0.0
        per_class_acc[f'class_{label_id}_accuracy'] = acc

    # Return metrics dictionary
    results = {
        'f1_macro': f1_macro,
    }
    results.update(per_class_acc)

    return results

if __name__ == "__main__":
    num_of_labels = 3

    parser = argparse.ArgumentParser(description="Fine-tune on course_eval.csv")
    parser.add_argument('--model', type=str, help="Model ID to load for finetuning. E.g. 'roberta'")
    parser.add_argument('--data', type=str, default="course_eval_raw.csv", help="CSV file path")
    args, unknown = parser.parse_known_args()

    # Load and process data
    tokenized_datasets = load_and_process_data(args.data)
    print(tokenized_datasets)

    # Load model
    if args.model == "roberta":
         model = RobertaForSequenceClassification.from_pretrained(
            model_id,
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
        learning_rate=1e-5,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=15,
        seed=42,
        save_steps=500,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    # Evaluate on the validation set (the 'test' split)
    results = trainer.evaluate()
    print("Validation Results:", results)

    # Save tokenizer and model
    # tokenizer.save_pretrained(f"{args.model}-aug-tokenizer")
    # trainer.save_model(f"{args.model}-agu-finetuned")
    # print("Model saved")