import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
import prompt_utils
import eda

from collections import Counter
from itertools import combinations
import random
random.seed(42)
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global classes 
classes = set()

def reformat_json_binary_v01(label, input_file):
    
    output_file_train = f"../data/labeled_data/lora_binary_train_{label.split('/')[0]}_v01.json"
    output_file_valid = f"../data/labeled_data/lora_binary_valid_{label.split('/')[0]}_v01.json"
    
    # Load both train and validation data
    with open(input_file, 'r') as f:
        data = json.load(f)

    positive_samples = []
    negative_samples = []
    
    # Combine train and validation data for processing
    combined_data = data["train"] + data["valid"]
    
    for item in combined_data:
        formatted_item = {
            "instruction": f"Classify the input based on if it's about {label}. Use 1 or 0 as output.",
            "input": normalizeTweet(item["text"]),
            "output": str(int(label in item["annotations"]))
        }
        if label in item["annotations"]:
            positive_samples.append(formatted_item)
        else:
            negative_samples.append(formatted_item)

    # Balance the positive and negative samples
    if len(negative_samples) > len(positive_samples):
        negative_samples = random.sample(negative_samples, len(positive_samples))

    # Merge positive and negative samples
    formatted_data = positive_samples + negative_samples

    print("---------------------")
    print("Label:", label)
    print("Positive samples:", len(positive_samples))
    print("Negative samples:", len(negative_samples))
    print("Total samples:", len(formatted_data))

    # Split the data into train and validation sets (80/20 ratio)
    train_data, valid_data = train_test_split(formatted_data, test_size=0.2, random_state=42)

    print("Train samples:", len(train_data))
    print("Valid samples:", len(valid_data))
    print("---------------------")

    return pd.DataFrame(train_data), pd.DataFrame(valid_data)

def co_occurrence(annotations):
    label_count = {}
    co_occur_count = {}
    
    for labels in annotations:
        for label in labels:
            label_count[label] = label_count.get(label, 0) + 1
            for co_label in labels:
                if label != co_label:
                    pair = tuple([label, co_label])
                    co_occur_count[pair] = co_occur_count.get(pair, 0) + 1

    co_occur_prob = {pair: count/label_count[pair[0]] for pair, count in co_occur_count.items()}

    return co_occur_prob

def inject_weak_labels(train_df, weak_labels_df, fsl_strategy):
    co_occur_prob = None
    if fsl_strategy == 'all':
        for index, row in weak_labels_df.iterrows():
            annotations = ast.literal_eval(row['annotations'])
            if annotations != []:
                train_df = train_df.append({'text': row['tweet_text'], 'annotations': annotations}, ignore_index=True)
    if fsl_strategy == 'distinct':
        for index, row in weak_labels_df.iterrows():
            annotations = ast.literal_eval(row['annotations'])
            #print(len(annotations))
            if len(annotations) == 1:
                train_df = train_df.append({'text': row['tweet_text'], 'annotations': annotations}, ignore_index=True)
    if fsl_strategy == 'co-occurrence':
        co_occur_prob = co_occurrence(train_df['annotations'])
        for index, row in weak_labels_df.iterrows():
            annotations = ast.literal_eval(row['annotations'])
            if len(annotations) == 1:
                train_df = train_df.append({'text': row['tweet_text'], 'annotations': annotations, 'co-occurence': True}, ignore_index=True)

    if fsl_strategy == 'eda':
        augmented_train_df = pd.DataFrame(columns=['text', 'annotations'])
        for index, row in train_df.iterrows():
            tweet_text = normalizeTweet(row['text'])
            augmented_tweets = eda.eda(tweet_text, alpha_sr=0.05, alpha_ri=0.05, alpha_rs=0.05, p_rd=0.5, num_aug=8)
            for augmented_tweet in augmented_tweets:
                augmented_train_df = augmented_train_df.append({'text': augmented_tweet, 'annotations': row['annotations']}, ignore_index=True)
        train_df = augmented_train_df
    
    return train_df, co_occur_prob

class BinaryDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def __call__(self, features: List[Dict[str, torch.Tensor]]):
        batch = super().__call__(features)
        batch["labels"] = torch.stack([feature["label"] for feature in features])
        return batch
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def binary_compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert logits to 0 or 1
    y_pred = np.argmax(predictions, axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, y_pred),
        "precision": precision_score(labels, y_pred),
        "recall": recall_score(labels, y_pred),
        "f1": f1_score(labels, y_pred),
    }

    return metrics

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
        print(logits)
        print(labels.float())
        return loss_fct(logits, labels.float())

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions, labels)
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

def normalizeTweet(tweet):
        tokens = TweetTokenizer().tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([normalizeToken(token) for token in tokens])

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

class TweetDataset(Dataset):
    def __init__(self, x, y, mlb, tokenizer, fsl_strategy = None, binary = False):
        self.x = x
        self.y = y
        self.mlb = mlb
        self.tokenizer = tokenizer
        self.max_length = 128
        self.encoded_tweets = self.preprocess_text(self.x, fsl_strategy)
        self.binary = binary

    def preprocess_text(self, X, fsl_strategy):
        X = [normalizeTweet(tweet) for tweet in X]
        
        return self.tokenizer(X, return_attention_mask=True, return_tensors='pt', padding=True, truncation = True, max_length=self.max_length)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.binary:
            label = self.y[idx].item()  # Convert tensor to a single integer value
            return {
                'input_ids': self.encoded_tweets['input_ids'][idx],
                'attention_mask': self.encoded_tweets['attention_mask'][idx],
                'label': torch.tensor(label)
            }
        else:
            label = self.y[idx]
            #print(label)
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
        
def main(task, epochs, train_size, validation_size, test_size, fp16, reporter, load_8bit, fsl_strategy, weak_labels_path, output_base, classification_type, class_name):
    # Train and evaluate your model using k-fold cross-validation
    k = 5
    results = {}
    if reporter == "None":
        reporter = []
    
    for i in range(3,5):
        
        print(f"Starting training of {i+1}. fold...")

        if output_base == None:
            output_base = "../models/"
            
        output_dir = f"{output_base}_{task}_epochs_{epochs}_train_size_{train_size}_fold_{i}"

        os.makedirs(output_dir, exist_ok=True)
        
        # Load the data for this fold
        filename = f"../data/labeled_data/{task}_test_{i}.json"

        if classification_type == "binary" and class_name != None:
            class_name = class_name.replace("_", " ")
            train_df, val_df = reformat_json_binary_v01(class_name, filename)

            with open(filename) as f:
                data = json.load(f)
            test_df = pd.DataFrame(data["test"])

            model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-large", load_in_8bit=load_8bit, num_labels=2, problem_type="single_label_classification")
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

            train_df['output'] = train_df['output'].astype(int)
            val_df['output'] = val_df['output'].astype(int)
            test_df['output'] = test_df['annotations'].apply(lambda x: 1 if class_name in x else 0)
            print(test_df['output'])
        
            train_dataset = TweetDataset(train_df['input'].to_list(), torch.tensor(train_df['output']), None, tokenizer, binary = True)
            val_dataset = TweetDataset(val_df['input'].to_list(), torch.tensor(val_df['output']), None, tokenizer, binary = True)
            test_dataset = TweetDataset(test_df['text'].to_list(), torch.tensor(test_df['output']), None, tokenizer, binary = True)
            data_collator = BinaryDataCollator(tokenizer)



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
                metric_for_best_model="f1",  # Adjusted for binary classification
                greater_is_better=True,
                save_total_limit=1,
                report_to=reporter,
            )

            # Create the Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=binary_compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Set patience to 3 because patience * eval_steps = 1,200
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
            return

        else:
            print(task)
            if task == "augmented":
                train_df = pd.read_csv(filename.replace(".json", ".csv"))
                with open(f"../data/labeled_data/generic_test_{i}.json") as f:
                    data = json.load(f)
                val_df = pd.DataFrame(data["valid"])
                test_df = pd.DataFrame(data["test"])
            else:
                with open(filename) as f:
                    data = json.load(f)
                train_df = pd.DataFrame(data["train"])
                val_df = pd.DataFrame(data["valid"])
                test_df = pd.DataFrame(data["test"])

            if (fsl_strategy != 'none') and (weak_labels_path != None):
                weak_labels_df = pd.read_csv(weak_labels_path)
                train_df, co_occurrence_probabilities = inject_weak_labels(train_df, weak_labels_df, fsl_strategy)
                train_df.to_csv(output_dir + '/train_df.csv', index=False)


            model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-large", load_in_8bit=load_8bit, num_labels=15, problem_type="multi_label_classification")
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

            if task == "augmented":
                train_df["annotations"] = train_df["annotations"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            train_annotations = train_df["annotations"].tolist()
            # Get all unique classes
            global classes
            classes = set()
            for annotation in train_annotations:
                classes.update(annotation)
            classes = sorted(list(classes))
            print("Classes: ", classes)

            # Convert the annotations to binary labels
            mlb = MultiLabelBinarizer(classes=classes)

            # train_size argument is used to control the size of the training set 
            if train_size != "full":
                train_df = train_df.sample(n=train_size)
            if validation_size != "full":
                val_df = val_df.sample(n=validation_size)
            if test_size != "full":
                test_df = test_df.sample(n=test_size)
        
            #print(co_occurrence_probabilities)
            train_labels = []
            for index, row in train_df.iterrows():
                try:
                    if row["co-occurence"] == True:
                        annotation = []
                        for class_ in classes:
                            if class_ == row["annotations"][0]:
                                annotation.append(1)
                            else:
                                try:
                                    annotation.append(co_occurrence_probabilities[tuple([row["annotations"][0], class_])])
                                except KeyError:
                                    annotation.append(0)
                        train_labels.append(annotation)
                    else:
                        train_labels.append(mlb.fit_transform([row["annotations"]])[0])
                except KeyError:
                    train_labels.append(mlb.fit_transform([row["annotations"]])[0])
                
            #train_labels = mlb.fit_transform(train_df["annotations"])
            val_labels = mlb.transform(val_df["annotations"])
            test_labels = mlb.transform(test_df["annotations"])
        
            train_dataset = TweetDataset(train_df['text'].to_list(), torch.tensor(train_labels), mlb, tokenizer, "eda")
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
                save_total_limit = 1,
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
    filename = f"../reports/{output_base}_{task}_epochs_{epochs}_train_size_{train_size}.json"
    
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
        
    tasks = ['generic', 'GRU_202012', 'IRA_202012', 'REA_0621', 'UGANDA_0621', 'VENEZUELA_201901_2', "augmented"]
    parser = argparse.ArgumentParser(description="Multilabel classification with k-fold cross-validation.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--train_size", type=int, default=None, help="Size of the training dataset (None for full dataset).")
    parser.add_argument("--validation_size", type=int, default=None, help="Size of the validation dataset (None for full dataset).")
    parser.add_argument("--test_size", type=int, default=None, help="Size of the test dataset (None for full dataset).")
    parser.add_argument("--task", type=str, choices=tasks, default='generic', help="One of 'generic', 'GRU_202012', 'IRA_202012', 'REA_0621', 'UGANDA_0621', 'VENEZUELA_201901', 'augmented'. ")
    parser.add_argument("--fp16", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use mixed precision training.")
    parser.add_argument("--reporter", type=str, default='wandb', choices=['wandb', 'None'], help="Use Weights and Biases for logging.")
    parser.add_argument("--load-8bit", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Train the model in 8bit. Use standalone --load-8bit to set to True.")
    # add fsl strategy (none, data augmentation (using EDA), all automatically labeled data, distinct classifications, balanced labels, forgiving labels, co-occurrence probabilities, majority_vote)
    parser.add_argument("--fsl-strategy", type=str, default='none', choices=['none', 'eda', 'all', 'distinct', 'balanced', 'forgiving', 'co-occurrence', 'majority_vote'], help="Few-shot learning strategy.")
    parser.add_argument("--weak-labels-path", type=str, default=None, help="Path to the weak labels.")
    parser.add_argument("--output-base-folder", type=str, default=None, help="Path to the output folder.")
    parser.add_argument("--classification-type", type=str, default="multi-label", help="One of 'binary' and 'multi-label'")
    parser.add_argument("--class-name", type=str, default=None, help="Class for binary classification")
    # for all automatically labeled data -> always inject training data with all labels. use 1/0 of the underrepresented labels to create [0, .., 0, 1,] labels
    # for distinct classifications -> use only the weak labels where every label is the only label for the tweet
    # for balanced labels -> add a balanced dataset to the training data
    # for forgiving labels -> add a forgiving labels [0.5, 0, 0, ..]
    # co-occurrence probabilities -> use the co-occurrence probabilities of each distinct classification
    # for majority vote -> use the weak labels of multiple models and use the majority vote as the label
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

    main(args.task, epochs, train_size, validation_size, test_size, args.fp16, args.reporter, args.load_8bit, args.fsl_strategy, args.weak_labels_path, args.output_base_folder, args.classification_type, args.class_name)
    
