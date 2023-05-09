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
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
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
from emoji import demojize
from nltk.tokenize import TweetTokenizer

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
    
    #TODO: check if without it, it trains
    @staticmethod
    def loss(logits, labels):
        # Use BCEWithLogitsLoss for multi-label classification
        loss_fct = torch.nn.BCEWithLogitsLoss()
        return loss_fct(logits, labels.float())

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1

    report = classification_report(labels, y_pred, labels=range(len(classes)), output_dict=True)

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
        self.max_length = 128
        self.encoded_tweets = self.preprocess_text(self.x)
        
    @staticmethod
    def normalizeToken(token):
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return token
    
    def normalizeTweet(self, tweet):
        tokens = TweetTokenizer().tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
                .replace("n't ", " n't ")
                .replace("n 't ", " n't ")
                .replace("ca n't", "can't")
                .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
                .replace("'re ", " 're ")
                .replace("'s ", " 's ")
                .replace("'ll ", " 'll ")
                .replace("'d ", " 'd ")
                .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
                .replace(" p . m ", " p.m ")
                .replace(" a . m .", " a.m.")
                .replace(" a . m ", " a.m ")
        )
        return " ".join(normTweet.split())

    def preprocess_text(self, X):
        X = [self.normalizeTweet(tweet) for tweet in X]
        
        return self.tokenizer(X, return_attention_mask=True, return_tensors='pt', padding=True, truncation = True, max_length=self.max_length)
        
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

def main(task, epochs, train_size, validation_size, test_size, fp16, reporter, load_8bit):
    # Train and evaluate your model using k-fold cross-validation
    k = 5
    results = {}
    if reporter == "None":
        reporter = []
    
    for i in range(k):
        
        print(f"Starting training of {i+1}. fold...")
        output_dir = f"./models/{task}_epochs_{epochs}_train_size_{train_size}_fold_{i}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the data for this fold
        filename = f"./data/labeled_data/{task}_test_{i}.json"
        with open(filename) as f:
            data = json.load(f)
        train_df = pd.DataFrame(data["train"])
        val_df = pd.DataFrame(data["valid"])
        test_df = pd.DataFrame(data["test"])
        
        model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-large", load_in_8bit=load_8bit ,num_labels=15, problem_type="multi_label_classification")
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
        
        # train_size argument is used to control the size of the training set 
        if train_size != "full":
            train_df = train_df.sample(n=train_size)
        if validation_size != "full":
            val_df = val_df.sample(n=validation_size)
        if test_size != "full":
            test_df = test_df.sample(n=test_size)
        
        train_labels = mlb.fit_transform(train_df["annotations"])
        val_labels = mlb.transform(val_df["annotations"])
        test_labels = mlb.transform(test_df["annotations"])
        
        train_dataset = TweetDataset(train_df['text'].to_list(), torch.tensor(train_labels), mlb, tokenizer)
        val_dataset = TweetDataset(val_df['text'].to_list(), torch.tensor(val_labels), mlb, tokenizer)
        test_dataset = TweetDataset(test_df['text'].to_list(), torch.tensor(test_labels), mlb, tokenizer)
        data_collator = MultiLabelDataCollator(tokenizer)

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
            weight_decay=0.0015,
            fp16=fp16,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            save_total_limit = 2,
            report_to=reporter,
        )

        # Create the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)], # Set patience to 3 because patience * eval_steps = 1,200
        )

        # start a new wandb run to track this script
        if reporter == "wandb":
            wandb.init(
                project="p6",
                config={
                "architecture": "Bertweet Large Base",
                "epochs": epochs,
                "task": task,
                "train_size": train_size,
                "setting": "weight-decay-0.0015, truncation=true, with maxlength"
                }
            )
        
        trainer.add_callback(CustomCallback(trainer)) 
        trainer.train()
        results[i] = {"valid": trainer.evaluate(), "test": trainer.evaluate(test_dataset)}
        trainer.save_model(output_dir)
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
    """ to consecutively train all campaigns:
    python ./src/reproduce_model.py --task generic --epochs 200
    python ./src/reproduce_model.py --task GRU_202012 --epochs 200
    python ./src/reproduce_model.py --task IRA_202012 --epochs 200
    python ./src/reproduce_model.py --task REA_0621 --epochs 200
    python ./src/reproduce_model.py --task UGANDA_0621 --epochs 200
    python ./src/reproduce_model.py --task VENEZUELA_201901_2 --epochs 200
    """
        
    tasks = ['generic', 'GRU_202012', 'IRA_202012', 'REA_0621', 'UGANDA_0621', 'VENEZUELA_201901_2']
    parser = argparse.ArgumentParser(description="Multilabel classification with k-fold cross-validation.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--train_size", type=int, default=None, help="Size of the training dataset (None for full dataset).")
    parser.add_argument("--validation_size", type=int, default=None, help="Size of the validation dataset (None for full dataset).")
    parser.add_argument("--test_size", type=int, default=None, help="Size of the test dataset (None for full dataset).")
    parser.add_argument("--task", type=str, choices=tasks, default='generic', help="One of 'generic', 'GRU_202012', 'IRA_202012', 'REA_0621', 'UGANDA_0621', 'VENEZUELA_201901' ")
    parser.add_argument("--fp16", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use mixed precision training.")
    parser.add_argument("--reporter", type=str, default='wandb', choices=['wandb', 'None'], help="Use Weights and Biases for logging.")
    parser.add_argument("--load-8bit", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Train the model in 8bit. Use standalone --load-8bit to set to True.")
    args = parser.parse_args()

    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 200
        
    if args.train_size:
        train_size = args.train_size
    else:
        train_size = "full"
        
    if args.validation_size:
        validation_size = args.validation_size
    else:
        validation_size = "full"
        
    if args.test_size:
        test_size = args.test_size
    else:
        test_size = "full"

    main(args.task, epochs, train_size, validation_size, test_size, args.fp16, args.reporter, args.load_8bit)
    
