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
import plotly.express as px
import math
from collections import defaultdict
import numpy as np
import matplotlib.colors as mcolors
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix, classification_report
import streamlit as st
# Set the default layout to wide mode
st.set_page_config(layout="wide")
import pandas as pd

classes = ["War/Terror", "Conspiracy Theory", "Education", "Election Campaign", "Environment", 
              "Government/Public", "Health", "Immigration/Integration", 
              "Justice/Crime", "Labor/Employment", 
              "Macroeconomics/Economic Regulation", "Media/Journalism", "Religion", "Science/Technology", "Others"]

def load_model(url, extraction_func, metrics_calculation_func):
    df = pd.read_csv(url)
    predictions_per_class, confusion_matrices, binary_classification_reports, multilabel_classification_reports = metrics_calculation_func(df, classes, extraction_func)
    data = {"confusion_matrices": confusion_matrices, "binary_classification_reports": binary_classification_reports, "multilabel_classification_reports": multilabel_classification_reports}
    return data, predictions_per_class

# ------------------------------
### Vicuna 4bit 
# ------------------------------

# Without context and classification only
## Example:
    ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\n### Assistant:\nClass = 
without_context_classification_only, without_context_classification_only_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_without_context_only_classification/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Without context and classification only
## Example:
    ### Human: Give the tweet a binary class based on if it's about {label} or not.\n\nTweet: {tweet_text}\n### Assistant:\nClass: "
without_context_classification_only_v02, without_context_classification_only_v02_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_without_context_only_classification_v02/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Without context and classification only
## Example:
    ### Human: Assign 1 if the tweet is about {label}. Assign 0 if it is not about {label}.\n\nTweet: {tweet_text}\n### Assistant:\nClass: "
without_context_classification_only_v03, without_context_classification_only_v03_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_without_context_only_classification_v03/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# without context and elaboration first
## Example:
    ### Human: Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\n### Assistant:\nElaboration: 
    #Followup :\n Assign class 1 for {label} or 0 for not. \n###Assistant:\nClass:  
without_context_elaboration_first, without_context_elaboration_first_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_without_context_elaboration_first/generic_test_0.csv", llm_utils.get_extraction_function("extract_last_float"), llm_utils.calculate_binary_metrics)

#with rules as context and classification only
## Example:
    ### Human: Based on rules, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nRules: {rules}\nTweet: {tweet_text}\n### Assistant:\nClass:
with_rules_classification_only, with_rules_classification_only_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_with_rules_only_classification/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 2), llm_utils.calculate_binary_metrics)

# with rules as context and elaboration first
## Example:
    ### Human: Based on rules, elaborate whether you think the Tweet is about {label}.\nRules: {rules}\nTweet: {tweet_text}\n### Assistant:\nElaboration: 
    #Followup :\n Assign class 1 for {label} or 0 for not. \n###Assistant:\nClass: 
with_rules_elaboration_first, with_rules_elaboration_first_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_with_rules_elaboration_first/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# without context and elaboration first v02
## Example:
    #prompt = f"### Human: Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\n### Assistant:\nElaboration: "
    #followup = f"### Human: Based on the elaboration, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nTweet: {tweet_text}\nElaboration: [ELABORATION]\n### Assistant:\nClass: "
without_context_elaboration_first_v02, without_context_elaboration_first_v02_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_without_context_elaboration_first_v02/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# without context and elaboration first v03
## Example:
    #prompt = f"### Human: Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\n### Assistant:\nElaboration: "
    #followup = f"### Human: Based on the elaboration, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nTweet: {tweet_text}\nElaboration: [ELABORATION]\n### Assistant:\nClass: "
without_context_elaboration_first_v03, without_context_elaboration_first_v03_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_without_context_elaboration_first_v03/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# without context and elaboration first v04
without_context_elaboration_first_v04, without_context_elaboration_first_v04_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_without_context_elaboration_first_v04/generic_test_0.csv", llm_utils.get_extraction_function("extract_using_class_token", 1), llm_utils.calculate_binary_metrics)

# with few-shot prompts
#----------------------
# Only classification 1 pos example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 1\n\nTweet: {tweet_text}\nClass: "
few_shot_1_pos, few_shot_1_pos_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_few_shot_prompt_only_classification_1_pos_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 1 neg example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
few_shot_1_neg, few_shot_1_neg_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_few_shot_prompt_only_classification_1_neg_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 1 random example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label}\n\nTweet: {tweet_text}\nClass: "
few_shot_1_random, few_shot_1_random_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_few_shot_prompt_only_classification_1_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 3 random example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass:
few_shot_3_random, few_shot_3_random_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_few_shot_prompt_only_classification_3_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 3 random example v02
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n Tweet: {example_tweet}\n### Assistant:\nClass: {example_tweet_label1}\nTweet: {example_tweet2}\n### Assistant:\nClass: {example_tweet_label2}\nTweet: {example_tweet3}\n### Assistant:\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\n### Assistant:\nClass: 
few_shot_3_random_v02, few_shot_3_random_v02_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_few_shot_prompt_only_classification_3_random_example_v02/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 5 random example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass:
few_shot_5_random, few_shot_5_random_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_few_shot_prompt_only_classification_5_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 10 random example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass:
few_shot_10_random, few_shot_10_random_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_few_shot_prompt_only_classification_10_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 1 pos 1 neg example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass:
few_shot_1_pos_1_neg, few_shot_1_pos_1_neg_predictions_per_class = load_model("../data/vicuna_4bit/generic_prompt_few_shot_prompt_only_classification_1_pos_1_neg_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# ------------------------------
### Vicuna 4bit LORA
# ------------------------------
vicuna_lora_multilabel_without_context_v01, vicuna_lora_multilabel_without_context_v01_predictions_per_class = load_model("../data/vicuna_4bit/lora/multilabel_without_context_v01/test_generic_test_0.csv", llm_utils.extract_multilabel_list, llm_utils.calculate_metrics_from_multilabel_list)
vicuna_lora_multilabel_without_context_v01_256_rank, vicuna_lora_multilabel_without_context_v01_256_rank_predictions_per_class = load_model("../data/vicuna_4bit/lora/multilabel_without_context_v01_256_rank/test_generic_test_0.csv", llm_utils.extract_multilabel_list, llm_utils.calculate_metrics_from_multilabel_list) 
vicuna_lora_multilabel_without_context_v01_retest, vicuna_lora_multilabel_without_context_v01_retest_predictions_per_class = load_model("../data/vicuna_4bit/lora/multilabel_without_context_v01_retest/test_generic_test_0.csv", llm_utils.extract_multilabel_list, llm_utils.calculate_metrics_from_multilabel_list)
vicuna_lora_multilabel_with_rules_v02, vicuna_lora_multilabel_with_rules_v02_predictions_per_class = load_model("../data/vicuna_4bit/lora/multilabel_with_rules_v02/test_generic_test_0.csv", llm_utils.extract_multilabel_list, llm_utils.calculate_metrics_from_multilabel_list)
vicuna_lora_multilabel_with_rules_v02_256_rank, vicuna_lora_multilabel_with_rules_v02_256_rank_predictions_per_class = load_model("../data/vicuna_4bit/lora/multilabel_with_rules_v02_256_rank/test_generic_test_0.csv", llm_utils.extract_multilabel_list, llm_utils.calculate_metrics_from_multilabel_list)

#-------------------------------
### Vicuna 4bit Multilabel In Context
#-------------------------------
vicuna_multi_label_no_fine_tune_explanation_first_v01, vicuna_multi_label_no_fine_tune_explanation_first_v01_predictions_per_class = load_model("../data/vicuna_4bit/multi_label_no_fine_tune_explanation_first_v01/test_generic_test_0.csv", llm_utils.extract_multilabel_list, llm_utils.calculate_metrics_from_multilabel_list)

# ------------------------------
### OpenAssistant LLama 30B 4bit
# ------------------------------

# without context and classification only
## Example:
    #f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nTweet: {tweet_text}\nClass: "
oa_without_context_classification_only, oa_without_context_classification_only_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_without_context_only_classification/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
oa_without_context_classification_only_v02, oa_without_context_classification_only_v02_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_without_context_only_classification_v02/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
oa_without_context_classification_only_v03, oa_without_context_classification_only_v03_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_without_context_only_classification_v03/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# without context and elaboration first
## Example:
    #"Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\nElaboration: "
    #Followup: \n\nAssign the label 1 for {label} or 0 for not.\nClass: 
oa_without_context_elaboration_first, oa_without_context_elaboration_first_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_without_context_elaboration_first/generic_test_0.csv", llm_utils.get_extraction_function("extract_label", 1), llm_utils.calculate_binary_metrics)
oa_without_context_elaboration_first_v02, oa_without_context_elaboration_first_v02_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_without_context_elaboration_first_v02/generic_test_0.csv", llm_utils.get_extraction_function("extract_label", 1), llm_utils.calculate_binary_metrics)
oa_without_context_elaboration_first_v04, oa_without_context_elaboration_first_v04_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_without_context_elaboration_first_v04/generic_test_0.csv", llm_utils.get_extraction_function("extract_using_class_token", 1), llm_utils.calculate_binary_metrics)

# Only classification 1 pos example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 1\n\nTweet: {tweet_text}\nClass: "
oa_few_shot_1_pos, oa_few_shot_1_pos_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_few_shot_prompt_only_classification_1_pos_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 1 neg example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
oa_few_shot_1_neg, oa_few_shot_1_neg_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_few_shot_prompt_only_classification_1_neg_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 1 random example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label}\n\nTweet: {tweet_text}\nClass: "
oa_few_shot_1_random, oa_few_shot_1_random_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_few_shot_prompt_only_classification_1_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 3 random example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass:
oa_few_shot_3_random, oa_few_shot_3_random_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_few_shot_prompt_only_classification_3_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 1 pos 1 neg example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass:
oa_few_shot_1_pos_1_neg, oa_few_shot_1_pos_1_neg_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_few_shot_prompt_only_classification_1_pos_1_neg_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 5 random example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass:
oa_few_shot_5_random, oa_few_shot_5_random_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_few_shot_prompt_only_classification_5_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# Only classification 10 random example
## ### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass:
oa_few_shot_10_random, oa_few_shot_10_random_predictions_per_class = load_model("../data/openassistant_llama_30b_4bit/generic_prompt_few_shot_prompt_only_classification_10_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# ------------------------------
### Openai GPT-3.5-turbo
# ------------------------------

# without context and classification only
## Example:
    #Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\nClass: 
gpt3_turbo_without_context_classification_only, gpt3_turbo_without_context_classification_only_predictions_per_class = load_model("../data/openai_gpt-3.5-turbo/generic_prompt_without_context_only_classification/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
gpt3_turbo_without_context_elaboration_first, gpt3_turbo_without_context_elaboration_first_predictions_per_class = load_model("../data/openai_gpt-3.5-turbo/generic_prompt_without_context_elaboration_first/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# ------------------------------
### Openai text-davinci-003
# ------------------------------

# without context and classification only
## Example:
    #Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\nClass: 
text_davinci_003_turbo_without_context_classification_only, text_davinci_003_turbo_without_context_classification_only_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_without_context_classification_only/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
## Example: 
    #f"Give the tweet a binary class based on if it's about {label} or not.\n\nTweet: {tweet_text}\nClass: "
text_davinci_003_turbo_without_context_classification_only_v02, text_davinci_003_turbo_without_context_classification_only_v02_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_without_context_classification_only_v02/generic_test_0.csv", llm_utils.get_extraction_function("extract_not_x", 2), llm_utils.calculate_binary_metrics)
## Example:
    #f"Assign 1 if the tweet is about {label}. Assign 0 if it is not about {label}.\n\nTweet: {tweet_text}\nClass: "
text_davinci_003_turbo_without_context_classification_only_v03, text_davinci_003_turbo_without_context_classification_only_v03_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_without_context_classification_only_v03/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# few shot prompt only classification
# 1 pos example
text_davinci_003_few_shot_1_pos, text_davinci_003_few_shot_1_pos_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_few_shot_prompt_only_classification_1_pos_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
text_davinci_003_few_shot_1_neg, text_davinci_003_few_shot_1_neg_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_few_shot_prompt_only_classification_1_neg_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
text_davinci_003_few_shot_3_random, text_davinci_003_few_shot_3_random_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_few_shot_prompt_only_classification_3_random_example/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# without context and elaboration first
## Example:
    #"Elaborate on whether you think the Tweet is about {label} or something else.\n\nTweet: {tweet_text}\n\n"
    #Followup: \nAssign the label 1 if it's about {label} or 0 for not based on the elaboration. Only output the number."
text_davinci_003_turbo_without_context_elaboration_first, text_davinci_003_turbo_without_context_elaboration_first_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_without_context_elaboration_first/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
text_davinci_003_turbo_without_context_elaboration_first_v02, text_davinci_003_turbo_without_context_elaboration_first_v02_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_without_context_elaboration_first_v02/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
text_davinci_003_turbo_without_context_elaboration_first_v03, text_davinci_003_turbo_without_context_elaboration_first_v03_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_without_context_elaboration_first_v03/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1, True), llm_utils.calculate_binary_metrics)

# with rules and classification only
## Example:
    #"Based on rules, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nRules: {rules}\nTweet: {tweet_text}\nClass: "
text_davinci_003_turbo_with_rules_classification_only, text_davinci_003_turbo_with_rules_classification_only_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_with_rules_classification_only/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# with rules and elaboration first
## Example:
    #prompt = f"Based on rules, elaborate whether you think the Tweet is about {label}.\nRules: {rules}\nTweet: {tweet_text}\nElaborations: "
    #followup = f"\nAssign the label 1 if it's about {label} or 0 for not based on the elaboration. Only output the number."
text_davinci_003_turbo_with_rules_elaboration_first, text_davinci_003_turbo_with_rules_elaboration_first_predictions_per_class = load_model("../data/openai_text_davinci_003/generic_prompt_with_rules_elaboration_first/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# ------------------------------
### GPT4xalpaca 4bit
# ------------------------------

# without context and elaboration first
## Example:
    ### Instruction:\nElaborate on whether you think the Tweet is about {label} or something else.\n\nTweet: {tweet_text}\n\n### Response:\n 
    #Followup: \n\n### Instruction:\nAssign the label 1 if you think it's about {label}, assign 0 if not.\n\n### Response:",
gpt4xalpaca_without_context_elaboration_first, gpt4xalpaca_without_context_elaboration_first_predictions_per_class = load_model("../data/gpt4xalpaca_4bit/generic_prompt_basic_elaboration_first/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)

# without context and classification only
## Example:
    #Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\n\n
gpt4xalpaca_without_context_classification_only, gpt4xalpaca_without_context_classification_only_predictions_per_class = load_model("../data/gpt4-x-alpaca_13b_4bit/generic_prompt_without_context_only_classification/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics) 
gpt4xalpaca_without_context_classification_only_v02, gpt4xalpaca_without_context_classification_only_v02_predictions_per_class = load_model("../data/gpt4-x-alpaca_13b_4bit/generic_prompt_without_context_only_classification_v02/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
gpt4xalpaca_without_context_classification_only_v03, gpt4xalpaca_without_context_classification_only_v03_predictions_per_class = load_model("../data/gpt4-x-alpaca_13b_4bit/generic_prompt_without_context_only_classification_v03/generic_test_0.csv", llm_utils.get_extraction_function("extract_nth_character", 1), llm_utils.calculate_binary_metrics)
gpt4xalpaca_without_context_elaboration_first_v04, gpt4xalpaca_without_context_elaboration_first_v04_predictions_per_class = load_model("../data/gpt4-x-alpaca_13b_4bit/generic_prompt_without_context_elaboration_first_v04/generic_test_0.csv", llm_utils.get_extraction_function("extract_using_class_token", 1), llm_utils.calculate_binary_metrics)

models = [
    {
        "model_name": "Vicuna 13B 4bit",
        "type": "Classification only V01",
        "context": "",
        "data": without_context_classification_only,
        "prediction_per_class": without_context_classification_only_predictions_per_class,
    },
        {
        "model_name": "Vicuna 13B 4bit",
        "type": "Classification only V02",
        "context": "",
        "data": without_context_classification_only_v02,
        "prediction_per_class": without_context_classification_only_v02_predictions_per_class,
    },
        {
        "model_name": "Vicuna 13B 4bit",
        "type": "Classification only V03",
        "context": "",
        "data": without_context_classification_only_v03,
        "prediction_per_class": without_context_classification_only_v03_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "",
        "type": "Elaboration first V01",
        "data": without_context_elaboration_first,
        "prediction_per_class": without_context_elaboration_first_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "",
        "type": "Elaboration first V02",
        "data": without_context_elaboration_first_v02,
        "prediction_per_class": without_context_elaboration_first_v02_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "",
        "type": "Elaboration first V03",
        "data": without_context_elaboration_first_v03,
        "prediction_per_class": without_context_elaboration_first_v03_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "",
        "type": "Elaboration first V04",
        "data": without_context_elaboration_first_v04,
        "prediction_per_class": without_context_elaboration_first_v04_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "With Rules",
        "type": "Classification only V01",
        "data": with_rules_classification_only,
        "prediction_per_class": with_rules_classification_only_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "With Rules",
        "type": "Elaboration first V01",
        "data": with_rules_elaboration_first,
        "prediction_per_class": with_rules_elaboration_first_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "1 pos example",
        "type": "Classification only",
        "data": few_shot_1_pos,
        "prediction_per_class": few_shot_1_pos_predictions_per_class,
    },

    {
        "model_name": "Vicuna 13B 4bit",
        "context": "1 random example",
        "type": "Classification only",
        "data": few_shot_1_random,
        "prediction_per_class": few_shot_1_random_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "1 neg example",
        "type": "Classification only",
        "data": few_shot_1_neg,
        "prediction_per_class": few_shot_1_neg_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "3 random examples",
        "type": "Classification only",
        "data": few_shot_3_random,
        "prediction_per_class": few_shot_3_random_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "3 random examples v02",
        "type": "Classification only",
        "data": few_shot_3_random_v02,
        "prediction_per_class": few_shot_3_random_v02_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "5 random examples",
        "type": "Classification only",
        "data": few_shot_5_random,
        "prediction_per_class": few_shot_5_random_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "10 random examples",
        "type": "Classification only",
        "data": few_shot_10_random,
        "prediction_per_class": few_shot_10_random_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit",
        "context": "1 pos 1 neg examples",
        "type": "Classification only",
        "data": few_shot_1_pos_1_neg,
        "prediction_per_class": few_shot_1_pos_1_neg_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit Multilabel LORA",
        "context": "",
        "type": "LoRA Multilabel Classification only",
        "data": vicuna_lora_multilabel_without_context_v01,
        "prediction_per_class": vicuna_lora_multilabel_without_context_v01_predictions_per_class,
    },
    {
        "model_name": "Vicuna 13B 4bit Multilabel LORA",
        "context": "",
        "type": "LoRA Multilabel Classification with 256 Rank",
        "data": vicuna_lora_multilabel_without_context_v01_256_rank,
        "prediction_per_class": vicuna_lora_multilabel_without_context_v01_256_rank_predictions_per_class,
    },

    {
        "model_name": "Vicuna 13B 4bit Multilabel LORA",
        "context": "",
        "type": "LoRA Multilabel Classification retest",
        "data": vicuna_lora_multilabel_without_context_v01_retest,
        "prediction_per_class": vicuna_lora_multilabel_without_context_v01_retest_predictions_per_class,
    },

    {
        "model_name": "Vicuna 13B 4bit Multilabel LORA",
        "context": "",
        "type": "LoRA Multilabel Classification with rules v02",
        "data": vicuna_lora_multilabel_with_rules_v02,
        "prediction_per_class": vicuna_lora_multilabel_with_rules_v02_predictions_per_class,
    },

    {
        "model_name": "Vicuna 13B 4bit Multilabel LORA",
        "context": "",
        "type": "LoRA Multilabel Classification with rules v02 and 256 Rank",
        "data": vicuna_lora_multilabel_with_rules_v02_256_rank,
        "prediction_per_class": vicuna_lora_multilabel_with_rules_v02_256_rank_predictions_per_class,
    },

    {
        "model_name": "Vicuna 13B 4bit Multilabel",
        "context": "",
        "type": "LoRA Multilabel In Context without fine tune explanation first v01",
        "data": vicuna_multi_label_no_fine_tune_explanation_first_v01,
        "prediction_per_class": vicuna_multi_label_no_fine_tune_explanation_first_v01_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "type": "Classification only V01",
        "context": "",
        "data": oa_without_context_classification_only,
        "prediction_per_class": oa_without_context_classification_only_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "type": "Classification only V02",
        "context": "",
        "data": oa_without_context_classification_only_v02,
        "prediction_per_class": oa_without_context_classification_only_v02_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "type": "Classification only V03",
        "context": "",
        "data": oa_without_context_classification_only_v03,
        "prediction_per_class": oa_without_context_classification_only_v03_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "type": "Elaboration first V01",
        "context": "",
        "data": oa_without_context_elaboration_first,
        "prediction_per_class": oa_without_context_elaboration_first_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "type": "Elaboration first V02",
        "context": "",
        "data": oa_without_context_elaboration_first_v02,
        "prediction_per_class": oa_without_context_elaboration_first_v02_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "type": "Elaboration first V04",
        "context": "",
        "data": oa_without_context_elaboration_first_v04,
        "prediction_per_class": oa_without_context_elaboration_first_v04_predictions_per_class,
    },
        {
        "model_name": "OA Llama 30B 4bit",
        "context": "1 pos example",
        "type": "Classification only",
        "data": oa_few_shot_1_pos,
        "prediction_per_class": oa_few_shot_1_pos_predictions_per_class,
    },

    {
        "model_name": "OA Llama 30B 4bit",
        "context": "1 random example",
        "type": "Classification only",
        "data": oa_few_shot_1_random,
        "prediction_per_class": oa_few_shot_1_random_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "context": "1 neg example",
        "type": "Classification only",
        "data": oa_few_shot_1_neg,
        "prediction_per_class": oa_few_shot_1_neg_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "context": "3 random examples",
        "type": "Classification only",
        "data": oa_few_shot_3_random,
        "prediction_per_class": oa_few_shot_3_random_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "context": "1 pos 1 neg examples",
        "type": "Classification only",
        "data": oa_few_shot_1_pos_1_neg,
        "prediction_per_class": oa_few_shot_1_pos_1_neg_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "context": "5 random examples",
        "type": "Classification only",
        "data": oa_few_shot_5_random,
        "prediction_per_class": oa_few_shot_5_random_predictions_per_class,
    },
    {
        "model_name": "OA Llama 30B 4bit",
        "context": "10 random examples",
        "type": "Classification only",
        "data": oa_few_shot_10_random,
        "prediction_per_class": oa_few_shot_10_random_predictions_per_class,
    },
    {
        "model_name": "Gpt 3.5-turbo",
        "type": "Classification only V01",
        "context": "",
        "data": gpt3_turbo_without_context_classification_only,
        "prediction_per_class": gpt3_turbo_without_context_classification_only_predictions_per_class,
    },
    {
        "model_name": "Gpt 3.5-turbo",
        "type": "Elaboration First V01",
        "context": "",
        "data": gpt3_turbo_without_context_elaboration_first,
        "prediction_per_class": gpt3_turbo_without_context_elaboration_first_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Classification only V01",
        "context": "",
        "data": text_davinci_003_turbo_without_context_classification_only,
        "prediction_per_class": text_davinci_003_turbo_without_context_classification_only_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Classification only v02",
        "context": "",
        "data": text_davinci_003_turbo_without_context_classification_only_v02,
        "prediction_per_class": text_davinci_003_turbo_without_context_classification_only_v02_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Classification only v03",
        "context": "",
        "data": text_davinci_003_turbo_without_context_classification_only_v03,
        "prediction_per_class": text_davinci_003_turbo_without_context_classification_only_v03_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Elaboration First V01",
        "context": "",
        "data": text_davinci_003_turbo_without_context_elaboration_first,
        "prediction_per_class": text_davinci_003_turbo_without_context_elaboration_first_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Elaboration First v02",
        "context": "",
        "data": text_davinci_003_turbo_without_context_elaboration_first_v02,
        "prediction_per_class": text_davinci_003_turbo_without_context_elaboration_first_v02_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Elaboration First v03",
        "context": "",
        "data": text_davinci_003_turbo_without_context_elaboration_first_v03,
        "prediction_per_class": text_davinci_003_turbo_without_context_elaboration_first_v03_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Classification Only",
        "context": "With Rules",
        "data": text_davinci_003_turbo_with_rules_classification_only,
        "prediction_per_class": text_davinci_003_turbo_with_rules_classification_only_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Elaboration First V01",
        "context": "With Rules",
        "data": text_davinci_003_turbo_with_rules_elaboration_first,
        "prediction_per_class": text_davinci_003_turbo_with_rules_elaboration_first_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Classification Only",
        "context": "1 pos example",
        "data": text_davinci_003_few_shot_1_pos,
        "prediction_per_class": text_davinci_003_few_shot_1_pos_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Classification Only",
        "context": "1 neg example",
        "data": text_davinci_003_few_shot_1_neg,
        "prediction_per_class": text_davinci_003_few_shot_1_neg_predictions_per_class,
    },
    {
        "model_name": "Text Davinci 003",
        "type": "Classification Only",
        "context": "3 random example",
        "data": text_davinci_003_few_shot_3_random,
        "prediction_per_class": text_davinci_003_few_shot_3_random_predictions_per_class,
    },
    {
        "model_name": "GPT4xalpaca 4bit",
        "type": "Elaboration first V01",
        "context": "",
        "data": gpt4xalpaca_without_context_elaboration_first,
        "prediction_per_class": gpt4xalpaca_without_context_elaboration_first_predictions_per_class,
    },
    {
        "model_name": "GPT4xalpaca 4bit",
        "type": "Classification only V01",
        "context": "",
        "data": gpt4xalpaca_without_context_classification_only,
        "prediction_per_class": gpt4xalpaca_without_context_classification_only_predictions_per_class,
    },
    {
        "model_name": "GPT4xalpaca 4bit",
        "type": "Classification only V02",
        "context": "",
        "data": gpt4xalpaca_without_context_classification_only_v02,
        "prediction_per_class": gpt4xalpaca_without_context_classification_only_v02_predictions_per_class,
    },
    {
        "model_name": "GPT4xalpaca 4bit",
        "type": "Classification only V03",
        "context": "",
        "data": gpt4xalpaca_without_context_classification_only_v03,
        "prediction_per_class": gpt4xalpaca_without_context_classification_only_v03_predictions_per_class,
    },
    {
        "model_name": "GPT4xalpaca 4bit",
        "type": "Classification only V04",
        "context": "",
        "data": gpt4xalpaca_without_context_elaboration_first_v04,
        "prediction_per_class": gpt4xalpaca_without_context_elaboration_first_v04_predictions_per_class,
    }

]

st.write(
    f"""
    <style>
        /* Style the sidebar */
        .sidebar .sidebar-content {{
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }}
        /* Style the radio buttons and add a hover effect */
        .stRadio label {{
            padding: 8px 16px;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 4px 0;
            transition: background-color 0.3s;
            width: 100% !important;
        }}
        .stRadio label:hover {{
            background-color: #e9ecef;
        }}
        # first label after class
        .stRadio:nth-child(1){{
            background-color: inherit !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar menu
page_options = ["Dashboard", "Box Plots", "Single Class Evaluation", "Multi-Class Evaluation", "Co-Occurrences"]
selected_page = st.sidebar.radio("Select Page", page_options)

# Title
st.title("Evaluation of LLMs")

if selected_page == "Single Class Evaluation":

    # Class selection
    class_options = classes
    selected_class = st.sidebar.selectbox("Select class", class_options)

    # Model selection
    model_options = [model["model_name"] + " " + model["context"] + " " + model["type"] for model in models]

    def enforce_max_selected(current_selection, max_selected=4):
        if len(current_selection) > max_selected:
            st.warning(f"Please select up to {max_selected} models.")
            return current_selection[:-1]  # Revert to the previous selection state
        return current_selection

    selected_models = st.sidebar.multiselect("Select Models (up to 4)", model_options, key="single_model_selection")
    selected_models = enforce_max_selected(selected_models)

    # Filter models based on user selection
    selected_models = [model for model in models if model["model_name"] + " " + model["context"] + " " + model["type"] in selected_models]

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
            }}
            .report-wrapper {{
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 0;
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
    if len(selected_models) == 0:
        st.warning("Please select at least one model from the sidebar.")
    else:
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
                if model['context'] == "":
                    st.header(f"{model['type']}")
                else:
                    st.header(f"{model['context']}, {model['type']}")

                # Add a custom div for the content with a fixed height and scrollbar
                st.write(
                    f"""
                    <div class="report-content">
                    """,
                    unsafe_allow_html=True,
                )

                selected_confusion_matrix = model["data"]["confusion_matrices"][selected_class]
                selected_classification_report = model["data"]["binary_classification_reports"][selected_class]

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

    # Class selection
    class_options = classes
    selected_class = st.sidebar.selectbox("Select class", class_options)

    # Model selection
    model_options = [model["model_name"] + " " + model["context"] + " " + model["type"] for model in models]

    def enforce_max_selected(current_selection, max_selected=4):
        if len(current_selection) > max_selected:
            st.warning(f"Please select up to {max_selected} models.")
            return current_selection[:-1]  # Revert to the previous selection state
        return current_selection

    selected_models = st.sidebar.multiselect("Select Models", model_options, key="multi_model_selection")
    #selected_models = enforce_max_selected(selected_models)

    # Filter models based on user selection
    selected_models = [model for model in models if model["model_name"] + " " + model["context"] + " " + model["type"] in selected_models]

    for idx, model in enumerate(selected_models):
        model_classification_reports_df = model["data"]["classification_reports"]
        #model_classification_reports_df = llm_utils.classification_reports_to_df(model_classification_reports, binary=True)

        st.write(
            f"""
            <div class="report-column">
            """,
            unsafe_allow_html=True,
        )
        # Display model name, type, and context in the header
        st.header(f"{model['model_name']}")
        if model['context'] == "":
            st.header(f"{model['type']}")
        else:
            st.header(f"{model['context']}, {model['type']}")

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

def calculate_fbeta_score(beta, precision, recall):
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

def calculate_metrics(model_classification_reports_df, beta):
    # Check if it's multilabel or binary classification
    low_f1_classes = ["Conspiracy Theory", "Education", "Environment", "Labor/Employment", "Science/Technology", "Religion"]
    if 'label' in model_classification_reports_df.columns:
        model_classification_reports_df = model_classification_reports_df.set_index('label')
    if 'f1_score_class_0' in model_classification_reports_df.columns:
        # Binary case
        avg_f1_class_0 = model_classification_reports_df['f1_score_class_0'].mean()
        avg_f1_class_1 = model_classification_reports_df['f1_score_class_1'].mean()
        avg_f1_score = (avg_f1_class_0 + avg_f1_class_1) / 2
        avg_f1_class_1_low = model_classification_reports_df.loc[low_f1_classes, 'f1_score_class_1'].mean()

        avg_accuracy = 0
        for idx, x in model_classification_reports_df.iterrows():
            try:
                true_positives = x['support_class_1'] * x['recall_class_1']
                true_negatives = x['support_class_0'] * x['recall_class_0']
                total_samples = x['support_class_0'] + x['support_class_1']
                accuracy = (true_positives + true_negatives) / total_samples
                avg_accuracy += accuracy
            except:
                pass
        avg_accuracy /= len(model_classification_reports_df)
        
        avg_precision_class_0 = model_classification_reports_df['precision_class_0'].mean()
        avg_precision_class_1 = model_classification_reports_df['precision_class_1'].mean()
        avg_recall_class_0 = model_classification_reports_df['recall_class_0'].mean()
        avg_recall_class_1 = model_classification_reports_df['recall_class_1'].mean()

        avg_precision_class_1_low = model_classification_reports_df.loc[low_f1_classes, 'precision_class_1'].mean()
        avg_recall_class_1_low = model_classification_reports_df.loc[low_f1_classes, 'recall_class_1'].mean()
        fbeta_score_class_1_low = calculate_fbeta_score(0.5, avg_precision_class_1_low, avg_recall_class_1_low)

        fbeta_score_class_0 = calculate_fbeta_score(beta, avg_precision_class_0, avg_recall_class_0)
        fbeta_score_class_1 = calculate_fbeta_score(beta, avg_precision_class_1, avg_recall_class_1)
        avg_fbeta_score = (fbeta_score_class_0 + fbeta_score_class_1) / 2

        return avg_f1_class_0, avg_f1_class_1, avg_f1_class_1_low, avg_f1_score, avg_accuracy, fbeta_score_class_0, fbeta_score_class_1, avg_fbeta_score, fbeta_score_class_1_low
    else:
    # Multilabel case
        avg_f1_scores = model_classification_reports_df.loc[low_f1_classes, 'f1-score'].tolist()
        avg_f1_score_low = sum(avg_f1_scores) / len(avg_f1_scores) if avg_f1_scores else None

        avg_precision_low = model_classification_reports_df.loc[low_f1_classes, 'precision'].tolist()
        avg_recall_low = model_classification_reports_df.loc[low_f1_classes, 'recall'].tolist()
        fbeta_scores_low = [calculate_fbeta_score(beta, precision, recall) for precision, recall in zip(avg_precision_low, avg_recall_low)]
        avg_fbeta_score_low = sum(fbeta_scores_low) / len(fbeta_scores_low) if fbeta_scores_low else None

        # The average scores for all classes
        avg_f1_score = model_classification_reports_df.loc['macro avg', 'f1-score']
        avg_accuracy = 0  # Not defined in the multilabel case
        avg_fbeta_score = calculate_fbeta_score(beta, model_classification_reports_df.loc['macro avg', 'precision'], model_classification_reports_df.loc['macro avg', 'recall'])
        
        return 0, 0, 0, avg_f1_score, avg_accuracy, 0, 0, avg_fbeta_score, avg_fbeta_score_low


def display_dashboard(models):
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
                color: #3c4043;
                font-weight: 500;
            }}
            .group-name {{
                text-align: center;
                font-weight: bold;
                margin-bottom: 8px;
                color: #3c4043;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Dashboard")

    cols = 3

    model_groups = defaultdict(list)
    for model in models:
        model_groups[model["model_name"]].append(model)

    # Add metric selection dropdown in the sidebar
    metric_options = ["None", "Avg F1-Score Class 0", "Avg F1-Score Class 1", "Avg-F1 Score", "Avg F1-Score Class 1 Low Class", "Avg F-Beta-Score Class 0", "Avg F-Beta-Score Class 1", "Avg F-Beta-Score", "Avg F-Beta-Score Low Class 1"]
    selected_metric = st.sidebar.radio("Select active metric", metric_options, index=1)

    # Prepare a color scale
    cmap = sns.color_palette("flare", as_cmap=True)

    for model_name, model_group in model_groups.items():
        st.write(f'<h2 class="group-name">{model_name}</h2>', unsafe_allow_html=True)

        num_models = len(model_group)
        rows = math.ceil(num_models / cols)

        row = 0
        col = 0
        outer_columns = st.columns(cols)

        # Collect metrics for color scale normalization
        beta = 0.5
        for idx, model in enumerate(model_group):
            if type(model["data"]["multilabel_classification_reports"]) != type(None):
                model_classification_reports_df = model["data"]["multilabel_classification_reports"]
            else:
                model_classification_reports_df = model["data"]["binary_classification_reports"]
            avg_f1_class_0, avg_f1_class_1, avg_f1_class_1_low, avg_f1_score, avg_accuracy, fbeta_score_class_0, fbeta_score_class_1, avg_fbeta_score, avg_fbeta_score_low = calculate_metrics(model_classification_reports_df, beta)

            metrics = {
                "Avg F1-Score Class 0": avg_f1_class_0, 
                "Avg F1-Score Class 1": avg_f1_class_1, 
                "Avg F1-Score": avg_f1_score, 
                "Avg F1-Score Class 1 Low Class": avg_f1_class_1_low,
                #"Avg Accuracy": avg_accuracy,
                "Avg F-Beta-Score Class 0": fbeta_score_class_0,
                "Avg F-Beta-Score Class 1": fbeta_score_class_1,
                "Avg F-Beta-Score": avg_fbeta_score,
                "Avg F-Beta-Score Low Class 1": avg_fbeta_score_low
            }
            selected_metric_value = metrics[selected_metric] * 1.3 - 0.7

            # Normalize metrics
            min_metric = 0.0
            max_metric = 1
            
            if selected_metric_value != "None":
                if max_metric != min_metric:
                    normalized_metric = (selected_metric_value - min_metric) / (max_metric - min_metric)
                else:
                    normalized_metric = 0.5  # Assign a neutral value when there is no variation in the metric

                background_color = mcolors.rgb2hex(cmap(normalized_metric))
            else:
                background_color = "None"

            with outer_columns[col]:
                if "multilabel" in model_name.lower():
                    st.write(#<div class="model-f1">Avg Accuracy: {avg_accuracy:.2f}</div>
                        f"""
                        <div class="model-box" style="background-color: {background_color}">
                            <div class="model-subtitle">{model['context']} {model['type']}</div>
                            <div class="model-f1">Avg F1-Score: {avg_f1_score:.2f}</div>
                            <div class="model-f1">Avg F-Beta-Score: {avg_fbeta_score:.2f}</div>
                            <div class="model-f1">Avg F-Beta-Score Low Class 1: {avg_fbeta_score_low:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(#<div class="model-f1">Avg Accuracy: {avg_accuracy:.2f}</div>
                        f"""
                        <div class="model-box" style="background-color: {background_color}">
                            <div class="model-subtitle">{model['context']} {model['type']}</div>
                            <div class="model-f1">Avg F1-Score Class 0: {avg_f1_class_0:.2f}</div>
                            <div class="model-f1">Avg F1-Score Class 1: {avg_f1_class_1:.2f}</div>
                            <div class="model-f1">Avg F1-Score: {avg_f1_score:.2f}</div>
                            <div class="model-f1">Avg F1-Score Class 1 Low Class: {avg_f1_class_1_low:.2f}</div>
                            <div class="model-f1">Avg F-Beta-Score Class 0: {fbeta_score_class_0:.2f}</div>
                            <div class="model-f1">Avg F-Beta-Score Class 1: {fbeta_score_class_1:.2f}</div>
                            <div class="model-f1">Avg F-Beta-Score: {avg_fbeta_score:.2f}</div>
                            <div class="model-f1">Avg F-Beta-Score Low Class 1: {avg_fbeta_score_low:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            col += 1
            if col == cols:
                col = 0
                row += 1
                if row < rows:
                    outer_columns = st.columns(cols)

def display_box_plots(models):
    

    st.write(
        f"""
        <style>
            .css-1v0mbdj img {{
                max-width: 1600px !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Box Plots")

    model_groups = defaultdict(list)
    for model in models:
        model_groups[model["model_name"]].append(model)

    metrics = ['f1_score_class_0', 'f1_score_class_1', 'average_f1_score', 'accuracy']

    all_models = []

    for metric in metrics:
        st.write(f'## {metric}')

        data_frames = []
        model_labels = []

        for model_name, model_group in model_groups.items():
            for model in model_group:
                model_classification_reports_df = model["data"]["classification_reports"]
                #model_classification_reports_df = llm_utils.classification_reports_to_df(model_classification_reports, binary=True)

                # Calculate average F1 score
                model_classification_reports_df['average_f1_score'] = (model_classification_reports_df['f1_score_class_0'] + model_classification_reports_df['f1_score_class_1']) / 2

                if metric == 'accuracy':
                    accuracies = []
                    for idx, x in model_classification_reports_df.iterrows():
                        try:
                            true_positives = x['support_class_1'] * x['recall_class_1']
                            true_negatives = x['support_class_0'] * x['recall_class_0']
                            total_samples = x['support_class_0'] + x['support_class_1']
                            accuracy = (true_positives + true_negatives) / total_samples
                            accuracies.append(accuracy)
                        except:
                            pass
                    metric_df = pd.DataFrame(accuracies, columns=[metric])
                else:
                    metric_df = model_classification_reports_df[[metric]].dropna()

                model_label = f"{model_name} {model['context']} {model['type']}"
                metric_df = metric_df.assign(Model=model_label)
                data_frames.append(metric_df)
                all_models.append(model_label)

        combined_df = pd.concat(data_frames, ignore_index=True)

        unique_models = sorted(list(set(all_models)))
        selected_models = st.multiselect(f"Select Models for {metric}", unique_models, default=unique_models, key=f'select_models_{metric}')

        filtered_df = combined_df[combined_df['Model'].isin(selected_models)]

        fig = px.box(filtered_df, x=metric, y='Model', labels={'Model': ''}, title=f'{metric} Boxplot')

        # Update the plot size
        fig.update_layout(width=1200, height=len(selected_models) * 40 + 200)

        st.plotly_chart(fig)

def parse_data(co_occurrence):
    data = []
    for key in co_occurrence:
        for inner_key in co_occurrence[key]:
            # get most common co-occurring labels
            most_common_co_occurring = co_occurrence[key][inner_key].most_common()
            for item in most_common_co_occurring:
                label, count = item  # unpack the tuple
                data.append([key, inner_key, label, count])
    return data

def visualize_data(data, col):
    # Convert the list to a DataFrame
    df = pd.DataFrame(data, columns=["Type", "Label", "CoLabel", "Count"])

    for category in df['Type'].unique():
        df_category = df[df['Type'] == category]
        
        # Pivot the DataFrame to the wide format
        df_pivot = df_category.pivot_table(index="Label", columns="CoLabel", values="Count", fill_value=0)

        # Plot a heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df_pivot, annot=True, cmap="YlGnBu")

        plt.title(f'({category})')

        # Display the plot in Streamlit
        with col:
            st.pyplot(fig)

import scipy.stats

def calculate_correlations(models, classes):
    # Calculate the confusion matrix for each model and each class
    confusion_matrices = {model["model_name"]: calculate_confusion_matrix(model, classes) for model in models}

    # Calculate the correlations between all confusion matrices
    for model1 in models:
        for model2 in models:
            if model1 != model2:
                correlations = {}
                for class_label in classes:
                    confusion_matrix1 = confusion_matrices[model1["model_name"]][class_label]
                    confusion_matrix2 = confusion_matrices[model2["model_name"]][class_label]
                    
                    # Flatten the confusion matrices into vectors
                    vector1 = confusion_matrix1.flatten()
                    vector2 = confusion_matrix2.flatten()
                    
                    # Calculate the Pearson and Spearman correlations
                    pearson_corr, _ = scipy.stats.pearsonr(vector1, vector2)
                    spearman_corr, _ = scipy.stats.spearmanr(vector1, vector2)
                    
                    correlations[class_label] = (pearson_corr, spearman_corr)
                
                print(f"Correlations between {model1['model_name']} and {model2['model_name']}:")
                for class_label, (pearson_corr, spearman_corr) in correlations.items():
                    print(f"  {class_label}: Pearson = {pearson_corr:.2f}, Spearman = {spearman_corr:.2f}")

if selected_page == "Co-Occurrences":

    # Model selection
    model_options = [model["model_name"] + " " + model["context"] + " " + model["type"] for model in models]
    selected_models = st.sidebar.multiselect("Select Models (up to 4)", model_options, key="single_model_selection")

    # Filter models based on user selection
    selected_models = [model for model in models if model["model_name"] + " " + model["context"] + " " + model["type"] in selected_models]

    # Display the results in the corresponding column
    if len(selected_models) == 0:
        st.warning("Please select at least one model from the sidebar.")
    else:
        columns = st.columns(2)  # Create 2 columns
        idx = 0  # Reset index
        #print(calculate_correlations(selected_models, classes))
        for model in selected_models:
            st.header(f"{model['model_name']}")
            if model['context'] == "":
                st.header(f"{model['type']}")
            else:
                st.header(f"{model['context']}, {model['type']}")

            
            co_occurrence = llm_utils.calculate_co_occurrence(model["prediction_per_class"], classes, verbose=True)
            data = parse_data(co_occurrence)
            visualize_data(data, columns[idx % 2])  # Choose column based on index
            idx += 1  # Increase index

if selected_page == "Dashboard":
    display_dashboard(models)
if selected_page == "Box Plots":
    display_box_plots(models)
