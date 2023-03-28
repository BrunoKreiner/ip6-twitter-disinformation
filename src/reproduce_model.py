import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainerCallback
)
from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict
from sklearn.metrics import classification_report

import argparse
import os
import ast
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict
import copy
import gc
import json
from tabulate import tabulate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global classes 
classes = set()

class MultiLabelDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def __call__(self, features: List[Dict[str, torch.Tensor]]):
        batch = super().__call__(features)
        batch["labels"] = torch.stack([feature["label"] for feature in features])
        return batch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        logits = model.classifier(pooled_output)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        # Remove the labels key from the inputs dictionary
        labels = inputs["labels"]
        inputs.pop("labels", None)

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = torch.mean(last_hidden_state, dim=1)
            logits = model.classifier(pooled_output)
        if labels is not None:
            return (None, logits, labels)
        else:
            return (None, logits, None)
        
    @staticmethod
    def loss(logits, labels):
        # Use BCEWithLogitsLoss for multi-label classification
        loss_fct = torch.nn.BCEWithLogitsLoss()
        return loss_fct(logits.view(-1, 15), labels.float().view(-1, 15))

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = (predictions > 0.5).astype(int)
    labels = labels.astype(int)
    report = classification_report(labels, predictions, labels=range(len(classes)), output_dict=True, zero_division=0)

    metrics = {
        "accuracy": np.mean(predictions == labels),
        "micro_precision": report["micro avg"]["precision"],
        "micro_recall": report["micro avg"]["recall"],
        "micro_f1": report["micro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
    }

    return metrics

class TweetDataset(Dataset):
    def __init__(self, x, y, mlb, tokenizer):
        self.x = x
        self.y = y
        self.mlb = mlb
        self.tokenizer = tokenizer
        self.encoded_tweets = self.preprocess_text(self.x)

    
    def preprocess_text(self, text):
        return self.tokenizer(text, return_attention_mask=True, return_tensors='pt', padding=True)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = self.y[idx]
        return {'input_ids': self.encoded_tweets['input_ids'][idx],
                'attention_mask': self.encoded_tweets['attention_mask'][idx],
                'label': torch.tensor(label, dtype=torch.float32)}
                #'label_ids': self.labels[idx]}

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def main(task, epochs, train_size):
    # Train and evaluate your model using k-fold cross-validation
    k = 5
    results = {}
    
    for i in range(k):
        print(i)
        output_dir = f"./models/{task}_epochs_{epochs}_train_size_{train_size}_fold_{i}"
        os.makedirs(output_dir, exist_ok=True)
        # Load the data for this fold
        filename = f"./data/labeled_data/{task}_test_{i}.json"
        with open(filename) as f:
            data = json.load(f)
        train_df = pd.DataFrame(data["train"])
        val_df = pd.DataFrame(data["valid"])
        test_df = pd.DataFrame(data["test"])
        
        # Load only a subset of the training data if train_size is specified
        if train_size != "full":
            train_df = train_df.sample(n=train_size)

        model = AutoModel.from_pretrained("vinai/bertweet-large", num_labels=15, problem_type="multi_label_classification")
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

        train_annotations = train_df["annotations"].tolist()

        # Get all unique classes
        global classes
        classes = set()
        for annotation in train_annotations:
            classes.update(annotation)
        
        classes = sorted(list(classes))

        # Convert the annotations to binary labels
        mlb = MultiLabelBinarizer(classes=classes)
        model.classifier = nn.Linear(model.config.hidden_size, len(classes))
        model.to(device)
        
        train_labels = mlb.fit_transform(train_df["annotations"])
        val_labels = mlb.transform(val_df["annotations"])
        test_labels = mlb.transform(test_df["annotations"])
        
        train_dataset = TweetDataset(train_df['text'].to_list(), torch.tensor(train_labels), mlb, tokenizer)
        val_dataset = TweetDataset(val_df['text'].to_list(), torch.tensor(val_labels), mlb, tokenizer)
        test_dataset = TweetDataset(test_df['text'].to_list(), torch.tensor(test_labels), mlb, tokenizer)
        data_collator = MultiLabelDataCollator(tokenizer)
        
        print(len(train_dataset))

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            eval_steps=400,
            save_steps=400,
            learning_rate=1e-5,
            weight_decay=0.001,
            fp16=True,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            save_total_limit = 2
        )

        # Create the Trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] # Set patience to 3 because patience * eval_steps = 1,200
        )

        # Train and evaluate your model on this fold
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="p6",
            
            # track hyperparameters and run metadata
            config={
            "architecture": "Bertweet Large Base",
            "epochs": epochs,
            "task": task,
            "train_size": train_size
            }
        )
        
        trainer.add_callback(CustomCallback(trainer)) 
        trainer.train()
        results[i] = {"valid": trainer.evaluate(), "test": trainer.evaluate(test_dataset)}
        
        del trainer
        del model
        del tokenizer
        
        torch.cuda.empty_cache()
        gc.collect()

    # Average the results across all k folds
    filename = f"./reports/{task}_epochs_{epochs}_train_size_{train_size}.json"
    
    # Check if the file exists and create it if it doesn't
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump(results, file)
    else:
        # Save dictionary to JSON file
        with open(filename, 'w') as file:
            json.dump(results, file)

    # Create list of dictionaries for tabulate
    table_data = []
    for k, v in results.items():
        row = {"Key": k}
        row.update(v)
        table_data.append(row)

    # Print table
    print(tabulate(table_data, headers="keys"))

if __name__ == "__main__":
    tasks = 'generic', 'GRU_202012', 'IRA_202012', 'REA_0621', 'UGANDA_0621', 'VENEZUELA_201901'
    parser = argparse.ArgumentParser(description="Multilabel classification with k-fold cross-validation.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--train_size", type=int, default=None, help="Size of the training dataset (None for full dataset).")
    parser.add_argument("--task", type=str, choices=tasks, default='generic', help="One of 'generic', 'GRU_202012', 'IRA_202012', 'REA_0621', 'UGANDA_0621', 'VENEZUELA_201901' ")
    args = parser.parse_args()

    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 200
        
    if args.train_size:
        train_size = args.train_size
    else:
        train_size = "full"
        
    
    main(args.task, epochs, train_size)