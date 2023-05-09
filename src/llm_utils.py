import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter

# Set display options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 15)

def classification_reports_to_df(classification_reports, binary):

    if binary:
        # Your code for creating the DataFrame and adding the results
        df = pd.DataFrame(columns=['label', 'f1_score_macro', 'precision_macro', 'recall_macro', 'support_macro',
                                        'f1_score_class_0', 'precision_class_0', 'recall_class_0', 'support_class_0',
                                        'f1_score_class_1', 'precision_class_1', 'recall_class_1', 'support_class_1'])

        for label, cr in classification_reports.items():
            try: 
                df = df.append({
                    'label': label,
                    'f1_score_macro': cr['macro avg']['f1-score'],
                    'precision_macro': cr['macro avg']['precision'],
                    'recall_macro': cr['macro avg']['recall'],
                    'support_macro': cr['macro avg']['support'],
                    'f1_score_class_0': cr['0']['f1-score'],
                    'precision_class_0': cr['0']['precision'],
                    'recall_class_0': cr['0']['recall'],
                    'support_class_0': cr['0']['support'],
                    'f1_score_class_1': cr['1']['f1-score'],
                    'precision_class_1': cr['1']['precision'],
                    'recall_class_1': cr['1']['recall'],
                    'support_class_1': cr['1']['support']
                }, ignore_index=True)
            except Exception as e:
                print(f"Error for {label}: {e}")
                df = df.append({
                    'label': label,
                    'f1_score_macro': None,
                    'precision_macro': None,
                    'recall_macro': None,
                    'support_macro': None,
                    'f1_score_class_0': None,
                    'precision_class_0': None,
                    'recall_class_0': None,
                    'support_class_0': None,
                    'f1_score_class_1': None,
                    'precision_class_1': None,
                    'recall_class_1': None,
                    'support_class_1': None
                }, ignore_index=True)
                continue

        # Display the results
        return df
    
    return None

def find_last_float(s):
    # Define the pattern for a float number
    float_pattern = r'\d+\.\d+|\d+'
    
    # Find all float numbers in the string
    floats = re.findall(float_pattern, s)

    # Reverse the list of found floats
    floats.reverse()
    
    # Extract the first element of the reversed list
    last_float = floats[0] if floats else None

    return last_float

def extract_label(classification_str):
    label_pattern = r"(?:Label:\s*)?(\d+)$"
    #print(classification_str)
    match = re.search(label_pattern, classification_str, re.MULTILINE)

    if match:
        #print(match.group(1))
        class_value = int(match.group(1))
        if class_value != 0 and class_value != 1:
            print("Class value not 0 or 1")
            print("---------------------")
            print(classification_str)
            print("----------------------")
            return None
        return class_value
    else:
        print("Label not found in the message: ", classification_str)
        return None

def extract_last_float(classification_str):
    if pd.isna(classification_str):
        return None
    try:
        class_value = float(find_last_float(classification_str))
        if class_value != 0 and class_value != 1:
            print("Class value not 0 or 1")
            print("---------------------")
            print(classification_str)
            print("----------------------")
            return None
        return class_value
    except ValueError:
        return None
    except IndexError:
        return None

# Function to convert 'Classification: [0 or 1]' string to int value
def extract_nth_character(classification_str, n):
    print(classification_str)
    if pd.isna(classification_str):
        return None
    if type(classification_str) == float:
        return classification_str
    try:
        #print(classification_str)
        class_value = int(classification_str[n])
        if class_value != 0 and class_value != 1:
            print("Class value not 0 or 1")
            print("---------------------")
            print(classification_str)
            print("----------------------")
            return None
        return class_value
    except ValueError:
        return None

def get_extraction_function(type, n=0):
    if type == "extract_nth_character":
        return lambda x: extract_nth_character(x, n-1)
    if type == "extract_last_float":
        return extract_last_float
    if type == "extract_label":
        return extract_label
    
def calculate_binary_metrics(df, classes, extraction_function):
    prediction_per_class = []
    # Iterate through class labels and extract binary predictions
    for idx, label in enumerate(classes):
        pred_column_name = f"{label}_pred"
        pred_column_df = df[df[pred_column_name].notna()].copy()
        pred_column_df[pred_column_name] = pred_column_df[pred_column_name].apply(extraction_function)
        prediction_per_class.append(pred_column_df)

    confusion_matrices = {}
    classification_reports = {}
    for idx, label in enumerate(classes):
        pred_column_name = f"{label}_pred"

        current_df = prediction_per_class[idx]
        
        # Ignore rows with NaN or invalid values in the predictions
        valid_rows = current_df[pred_column_name].notna()
        
        y_true = current_df.loc[valid_rows, 'annotations'].apply(lambda x: int(label in x))
        y_pred = current_df.loc[valid_rows, pred_column_name].astype(int)

        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices[label] = cm
        cr = classification_report(y_true, y_pred, output_dict=True)
        classification_reports[label] = cr

    return prediction_per_class, confusion_matrices, classification_reports

def calculate_co_occurrence(prediction_per_class, classes, ignore_same_label=True, verbose = False):

    def update_co_occurrence_dict(co_occurrence, y_true, y_pred, label, annotations):
        for i in range(len(y_true)):
            co_occuring_labels_raw = set(annotations.iloc[i].split(", "))
            if ignore_same_label:
                co_occuring_labels = {lbl.strip("[]'") for lbl in co_occuring_labels_raw if lbl.strip("[]'") != label}
            else:
                co_occuring_labels = {lbl.strip("[]'") for lbl in co_occuring_labels_raw}
            
            if y_true.iloc[i] == 1 and y_pred.iloc[i] == 1:
                co_occurrence["tp"][label].update(co_occuring_labels)
            elif y_true.iloc[i] == 1 and y_pred.iloc[i] == 0:
                co_occurrence["fn"][label].update(co_occuring_labels)
            elif y_true.iloc[i] == 0 and y_pred.iloc[i] == 1:
                co_occurrence["fp"][label].update(co_occuring_labels)
            else:
                co_occurrence["tn"][label].update(co_occuring_labels)
        return co_occurrence

    # Initialize co_occurrence dictionary
    co_occurrence = {"tp": defaultdict(Counter), "fp": defaultdict(Counter), "tn": defaultdict(Counter), "fn": defaultdict(Counter)}

    # Calculate co_occurrence
    for idx, label in enumerate(classes):
        pred_column_name = f"{label}_pred"

        current_df = prediction_per_class[idx]

        valid_rows = current_df[pred_column_name].notna()

        y_true = current_df.loc[valid_rows, 'annotations'].apply(lambda x: int(label in x))
        y_pred = current_df.loc[valid_rows, pred_column_name].astype(int)
        annotations = current_df.loc[valid_rows, 'annotations']

        co_occurrence = update_co_occurrence_dict(co_occurrence, y_true, y_pred, label, annotations)

    # Print most co-occurring labels
    print("Most co-occurring labels in true positives, false positives, true negatives, and false negatives:")
    for key, value in co_occurrence.items():
        print(f"{key}:")
        for label, counter in value.items():
            most_common_co_occurring = counter.most_common()
            print(f"  {label}: {most_common_co_occurring}")
        print()

    return co_occurrence


    
# Function to assign 'Others' label if none of the prediction columns have a 1
def assign_others_to_row(row, classes):
    if not any(row[f"{label}_pred"] == 1 for label in classes):
        return 1
    return 0

def print_confusion_matrix(classification_reports):
    for label, cm in classification_reports["confusion_matrices"].items():
        print(f"Confusion matrix for {label}:")
        print(cm)
        print()

"""test_df['Others_pred'] = test_df.apply(assign_others_to_row, axis=1)
classes.append('Others')"""
        