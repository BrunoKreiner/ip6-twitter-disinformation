import sys
import os
sys.path.append("../src")
import llm_utils
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import math
from collections import defaultdict

from sklearn.metrics import confusion_matrix, classification_report

import streamlit as st

# Set the default layout to wide mode
st.set_page_config(layout="wide")
import pandas as pd


classes = ["War/Terror", "Conspiracy Theory", "Education", "Election Campaign", "Environment", 
              "Government/Public", "Health", "Immigration/Integration", 
              "Justice/Crime", "Labor/Employment", 
              "Macroeconomics/Economic Regulation", "Media/Journalism", "Religion", "Science/Technology"]

# ------------------------------
### Vicuna 4bit 
# ------------------------------

# Without context and classification only
## Example:
    ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\n### Assistant:\nClass = 
without_context_classification_only_df = pd.read_csv("../data/vicuna_4bit/generic_prompt_without_context_only_classification/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(without_context_classification_only_df, classes, extraction_function)
without_context_classification_only = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

# without context and elaboration first
## Example:
    ### Human: Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\n### Assistant:\nElaboration: 
    #Followup :\n Assign class 1 for {label} or 0 for not. \n###Assistant:\nClass:  
without_context_elaboration_first_df = pd.read_csv("../data/vicuna_4bit/generic_prompt_without_context_elaboration_first/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_last_float")
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(without_context_elaboration_first_df, classes, extraction_function)
without_context_elaboration_first = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

#with rules as context and classification only
## Example:
    ### Human: Based on rules, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nRules: {rules}\nTweet: {tweet_text}\n### Assistant:\nClass:
with_rules_classification_only_df = pd.read_csv("../data/vicuna_4bit/generic_prompt_with_rules_only_classification/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 2)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(with_rules_classification_only_df, classes, extraction_function)
with_rules_classification_only = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

# with rules as context and elaboration first
## Example:
    ### Human: Based on rules, elaborate whether you think the Tweet is about {label}.\nRules: {rules}\nTweet: {tweet_text}\n### Assistant:\nElaboration: 
    #Followup :\n Assign class 1 for {label} or 0 for not. \n###Assistant:\nClass: 
with_rules_elaboration_first_df = pd.read_csv("../data/vicuna_4bit/generic_prompt_with_rules_elaboration_first/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(with_rules_elaboration_first_df, classes, extraction_function)
with_rules_elaboration_first = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

# ------------------------------
### OpenAssistant LLama 30B 4bit
# ------------------------------

# without context and classification only
## Example:
    #f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nTweet: {tweet_text}\nClass: "

oa_without_context_classification_only_df = pd.read_csv("../data/openassistant_llama_30b_4bit/generic_prompt_without_context_only_classification/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(oa_without_context_classification_only_df, classes, extraction_function)
oa_without_context_classification_only = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}


# without context and elaboration first
## Example:
    #"Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\nElaboration: "
    #Followup: \n\nAssign the label 1 for {label} or 0 for not.\nClass: 

oa_without_context_elaboration_first_df = pd.read_csv("../data/openassistant_llama_30b_4bit/generic_prompt_without_context_elaboration_first_v02/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_label", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(oa_without_context_elaboration_first_df, classes, extraction_function)
oa_without_context_elaboration_first = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

# ------------------------------
### Openai GPT-3.5-turbo
# ------------------------------

# without context and classification only
## Example:
    #Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\nClass: 

gpt3_turbo_without_context_classification_only_df = pd.read_csv("../data/openai_gpt-3.5-turbo/generic_prompt_without_context_only_classification/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(gpt3_turbo_without_context_classification_only_df, classes, extraction_function)
gpt3_turbo_without_context_classification_only = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}


# ------------------------------
### Openai text-davinci-003
# ------------------------------

# without context and classification only
## Example:
    #Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\nClass: 

text_davinci_003_turbo_without_context_classification_only_df = pd.read_csv("../data/openai_text_davinci_003/generic_prompt_without_context_classification_only/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(text_davinci_003_turbo_without_context_classification_only_df, classes, extraction_function)
text_davinci_003_turbo_without_context_classification_only = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

# without context and elaboration first
## Example:
    #"Elaborate on whether you think the Tweet is about {label} or something else.\n\nTweet: {tweet_text}\n\n"
    #Followup: \nAssign the label 1 if it's about {label} or 0 for not based on the elaboration. Only output the number."

text_davinci_003_turbo_without_context_elaboration_first_df = pd.read_csv("../data/openai_text_davinci_003/generic_prompt_without_context_elaboration_first/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(text_davinci_003_turbo_without_context_elaboration_first_df, classes, extraction_function)
text_davinci_003_turbo_without_context_elaboration_first = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

# ------------------------------
### GPT4xalpaca 4bit
# ------------------------------

# without context and elaboration first
## Example:
    ### Instruction:\nElaborate on whether you think the Tweet is about {label} or something else.\n\nTweet: {tweet_text}\n\n### Response:\n 
    #Followup: \n\n### Instruction:\nAssign the label 1 if you think it's about {label}, assign 0 if not.\n\n### Response:",

gpt4xalpaca_without_context_elaboration_first_df = pd.read_csv("../data/openai_text_davinci_003/generic_prompt_without_context_classification_only/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(gpt4xalpaca_without_context_elaboration_first_df, classes, extraction_function)
gpt4xalpaca_without_context_elaboration_first = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

# without context and classification only
## Example:
    #Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\n\n

gpt4xalpaca_without_context_classification_only_df = pd.read_csv("../data/openai_text_davinci_003/generic_prompt_without_context_classification_only/generic_test_0.csv")
extraction_function = llm_utils.get_extraction_function("extract_nth_character", 1)
prediction_per_class, confusion_matrices, classification_reports = llm_utils.calculate_binary_metrics(gpt4xalpaca_without_context_classification_only_df, classes, extraction_function)
gpt4xalpaca_without_context_classification_only = {"confusion_matrices": confusion_matrices, "classification_reports": classification_reports}

models = [
    {
        "model_name": "Vicuna 13B 4bit",
        "type": "Classification only",
        "context": "No context",
        "data": without_context_classification_only,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "No context",
        "type": "Elaboration first",
        "data": without_context_elaboration_first,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "With Rules",
        "type": "Classification only",
        "data": with_rules_classification_only,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "With Rules",
        "type": "Elaboration first",
        "data": with_rules_elaboration_first,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "type": "Classification only",
        "context": "No context",
        "data": oa_without_context_classification_only,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "type": "Elaboration first",
        "context": "No context",
        "data": oa_without_context_elaboration_first,
    },
    {
        "model_name": "Gpt 3.5-turbo",
        "type": "Classification only",
        "context": "No context",
        "data": gpt3_turbo_without_context_classification_only,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Classification only",
        "context": "No context",
        "data": text_davinci_003_turbo_without_context_classification_only,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Elaboration First",
        "context": "No context",
        "data": text_davinci_003_turbo_without_context_elaboration_first,
    },
    {
        "model_name": "GPT4xalpaca 4bit",
        "type": "Elaboration first",
        "context": "No context",
        "data": gpt4xalpaca_without_context_elaboration_first,
    },
    {
        "model_name": "GPT4xalpaca 4bit",
        "type": "Classification only",
        "context": "No context",
        "data": gpt4xalpaca_without_context_classification_only,
    }

]

# Sidebar menu
page_options = ["Overview", "Single Class Evaluation", "Multi-Class Evaluation"]
selected_page = st.sidebar.radio("Select Page", page_options)

# Title
st.title("Evaluation of LLMs")

# Class selection
class_options = classes
selected_class = st.sidebar.selectbox("Select class", class_options)

# Model selection
model_options = [model["model_name"] + " " + model["context"] + " " + model["type"] for model in models]
selected_model_1 = st.sidebar.selectbox("Select Model 1", model_options)
selected_model_2 = st.sidebar.selectbox("Select Model 2", model_options)
selected_model_3 = st.sidebar.selectbox("Select Model 3", model_options)
selected_model_4 = st.sidebar.selectbox("Select Model 4", model_options)

# Filter models based on user selection
print(selected_model_1)
selected_models = [model for model in models if model["model_name"] + " " + model["context"] + " " + model["type"] in [selected_model_1, selected_model_2, selected_model_3, selected_model_4]]

if selected_page == "Single Class Evaluation":

    # Inject custom CSS
    st.write(
        f"""
        <style>
            .css-ocqkz7 {{
                display: flex;
                flex-wrap: wrap;
                -webkit-box-flex: 1;
                flex-grow: 1;
                -webkit-box-align: end;
                align-items: flex-end;
                gap: 1rem;
            }}
            .report-wrapper {{
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 0;
            }}
            .report-column {{
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }}
            .report-content {{
                overflow-y: auto;
            }}
            .css-1v0mbdj img {{
                max-width: 300px !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Wrap the columns in a custom div to center them vertically
    st.write('<div class="report-wrapper">', unsafe_allow_html=True)

    columns = st.beta_columns(len(selected_models))
    for idx, model in enumerate(selected_models):
        # Display the results in the corresponding column
        with columns[idx]:
            st.write(
                f"""
                <div class="report-column">
                """,
                unsafe_allow_html=True,
            )
            # Display model name, type, and context in the header
            st.header(f"{model['model_name']}")
            if model['context'] == "No context":
                st.header(f"{model['type']}")
            else:
                st.header(f"{model['type']}, {model['context']}")

            # Add a custom div for the content with a fixed height and scrollbar
            st.write(
                f"""
                <div class="report-content">
                """,
                unsafe_allow_html=True,
            )

            selected_confusion_matrix = model["data"]["confusion_matrices"][selected_class]
            selected_classification_report = model["data"]["classification_reports"][selected_class]

            # Display the confusion matrix using Seaborn's heatmap
            st.subheader("Confusion Matrix")
            fig = plt.figure(figsize=(3, 3))
            sns.heatmap(selected_confusion_matrix, annot=True, fmt='d', cmap='coolwarm')
            st.pyplot(fig)

            # Display the classification report as a simple list
            st.subheader("Classification Report")
            report_df = pd.DataFrame(selected_classification_report).transpose().drop("weighted avg")
            for row in report_df.itertuples():
                st.subheader(row.Index)
                for col, val in zip(report_df.columns, row[1:]):
                    if row.Index == "accuracy":
                        st.write(f'{val:.2f}')
                        break
                    else:
                        st.write(f'{col}: {val:.2f}')

            # Close the custom div for the content
            st.write(
                f"""
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Close the custom div for the wrapper
    st.write('</div>', unsafe_allow_html=True)

if selected_page == "Multi-Class Evaluation":

    # Filter models based on user selection
    selected_models = [model for model in models if model["model_name"] + " " + model["context"] + " " + model["type"] in [selected_model_1, selected_model_2, selected_model_3, selected_model_4]]

    for idx, model in enumerate(selected_models):
        model_classification_reports = model["data"]["classification_reports"]
        model_classification_reports_df = llm_utils.classification_reports_to_df(model_classification_reports, binary=True)

        st.write(
            f"""
            <div class="report-column">
            """,
            unsafe_allow_html=True,
        )
        # Display model name, type, and context in the header
        st.header(f"{model['model_name']}")
        if model['context'] == "No context":
            st.header(f"{model['type']}")
        else:
            st.header(f"{model['type']}, {model['context']}")

        # Display the classification report for all classes
        st.subheader("Classification Report for All Classes")
        st.write(model_classification_reports_df)

        # Close the custom div for the content
        st.write(
            f"""
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def make_grid(cols, rows):
    grid = [0] * cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

if selected_page == "Overview":
    st.write(
        f"""
        <style>
            .model-grid-container {{
                display: flex !important;
                flex-wrap: wrap !important;
                justify-content: center !important;
                align-items: center !important;
            }}
            .model-grid {{
                display: grid !important;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)) !important;
                gap: 1rem !important;
            }}
            .model-box {{
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                background-color: #f8f9fa;
                margin: 1rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
            }}
            .model-box:hover {{
                transform: translateY(-10px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .model-name {{
                font-size: 1.25rem;
                font-weight: bold;
                color: #3c4043;
                margin-bottom: 0.5rem;
                text-align: center;
            }}
            .model-f1 {{
                font-size: 1rem;
                color: #5a6268;
            }}
            .group-name {{
                text-align: center;
                font-weight: bold;
                margin-bottom: 8px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Dashboard")

    cols = 3

    # Group the models by model_name
    model_groups = defaultdict(list)
    for model in models:
        model_groups[model["model_name"]].append(model)

    # Iterate over the model groups and create a grid for each group
    for model_name, model_group in model_groups.items():
        st.write(f'<h2 class="group-name">{model_name}</h2>', unsafe_allow_html=True)

        # Calculate the number of rows based on the number of models and columns
        num_models = len(model_group)
        rows = math.ceil(num_models / cols)

        # Create the grid
        row = 0
        col = 0
        outer_columns = st.columns(cols)

        for idx, model in enumerate(model_group):
            model_classification_reports = model["data"]["classification_reports"]
            model_classification_reports_df = llm_utils.classification_reports_to_df(model_classification_reports, binary=True)
            avg_f1_macro = model_classification_reports_df['f1_score_macro'].mean()
            #avg_f1_micro_weighted = model_classification_reports_df['f1_score_micro_weighted'].mean()

            with outer_columns[col]:
                st.write(
                    f"""
                    <div class="model-box">
                        <div class="model-subtitle">{model['type']} {model['context']}</div>
                        <div class="model-f1">Avg F1 Score Macro: {avg_f1_macro:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Update column and row indices
            col += 1
            if col == cols:
                col = 0
                row += 1
                if row < rows:
                    outer_columns = st.columns(cols)
