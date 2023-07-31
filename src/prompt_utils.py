from emoji import demojize
from nltk.tokenize import TweetTokenizer
import requests
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer
from datetime import datetime
import pandas as pd
import os
import openai

HOST = 'http://127.0.0.1:5000'
URI = f'{HOST}/api/v1/generate'
RULES = ["Oxford dictionary's definition of war: “situation in which two or more countries or groups of people fight against each other over a period of time”. Oxford dictionary's definition of terror (terrorism): “violent action or the threat of violent action that is intended to cause fear, usually for political purposes”. Remark: This category includes also causes and consequences of war/terror (e.g. “the current situation in Ukraine may cause a supply crisis for wheat products”).",
"Oxford dictionary's definition of conspiracy: “a secret plan by a group of people to do something harmful or illegal”. Remark: Assignment of this category may depend on viewpoint and political stance of rater, which can be mitigated by focusing on the definition above. If the content of a tweet describes a conspiratorial activity/process, it will be labeled “conspiracy theory”.",
"Oxford dictionary's definition of education: “a process of teaching, training and learning, especially in schools, colleges or universities, to improve knowledge and develop skills”. Remark: Does not include education/training of soldiers (🡪war/terror).",
"Oxford dictionary's definition of election: “the process of choosing a person or a group of people for a position, especially a political position, by voting”. Remark: This category includes all activities aimed at rallying the population for participation in a public election, description of election outcomes, and conduct of the election itself.",
"Oxford dictionary's definition of environment: “the natural world in which people, animals and plants live”. Remark: This category is typically used for tweet content revolving around activities and processes affecting the environment in some way.",
"Oxford dictionary's definition of government: “the group of people who are responsible for controlling a country or a state”. Oxford dictionary's definition of public: “ordinary people who are not members of a particular group or organization” Remark: This category includes also statements/content about the public perception of activities/processes of government (i.e. voiced criticism or praise for a government).",
"Oxford dictionary's definition of health: “the condition of a person's body or mind”. Remark: This category includes also statements related to public health. In such a case both Health and Government/Public must be selected.",
"Oxford dictionary's definition of immigration: “the process of coming to live permanently in a different country from the one you were born in”. Oxford dictionary's definition of integration: “the act or process of mixing people who have previously been separated, usually because of colour, race, religion, etc.”",
"Oxford dictionary's definition of justice: “the legal system used to punish people who have committed crimes”. Oxford dictionary's definition of crime: “activities that involve breaking the law”. Remark: This category does not include statements/content on war crimes (🡪 war/terror).",
"Oxford dictionary's definition of labor: “work, especially physical work”. Oxford dictionary's definition of employment: “work, especially when it is done to earn money; the state of being employed”.",
"Oxford dictionary's definition of macroeconomics: “the study of large economic systems, such as those of whole countries or areas of the world”. Oxford dictionary's definition of regulation: ”an official rule made by a government or some other authority”. Remark: In case of statements/content on economic regulations, this category may likely co-occur with Government/Public category.", 
"Oxford dictionary's definition of media: “the main ways that large numbers of people receive information and entertainment, that is television, radio, newspapers and the internet”. Oxford dictionary's definition of journalism: “the work of collecting and writing news stories for newspapers, magazines, radio, television or online news sites; the news stories that are written”. Remark: This category will be used for statements/content which explicitly references other media outlets or journalists (e.g. “BBC has reported that …”, “Bellingcat has discovered a secret operation of X”). Content which appears “news-worthy” does not generally fall into this category (🡪 newsworthiness is very subjective and context-dependent).",
"Oxford dictionary's definition of religion: “the belief in the existence of a god or gods, and the activities that are connected with the worship of them, or in the teachings of a spiritual leader”.",
"Oxford dictionary's definition of science: “knowledge about the structure and behavior of the natural and physical world, based on facts that you can prove, for example by experiments”. Oxford dictionary's definition of technology: “scientific knowledge used in practical ways in industry, for example in designing new machines”."]
ALL_LABELS = all_labels = ["War/Terror", "Conspiracy Theory", "Education", "Election Campaign", "Environment", 
              "Government/Public", "Health", "Immigration/Integration", 
              "Justice/Crime", "Labor/Employment", 
              "Macroeconomics/Economic Regulation", "Media/Journalism", "Religion", "Science/Technology", "Others"]
LOW_F1_LABELS = ["Conspiracy Theory", "Education", "Environment", "Labor/Employment", "Religion", "Science/Technology"]
openai.api_key = "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"

def normalize_token_simplified(token):
    """normalize token function as opposed to:
    -------------------------------------------
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
    -------------------------------------------
    Args:
        token (string): token to normalize

    Returns:
        str: normalized token
    """
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "[url]"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token
    
def normalize_tweet_simplified(tweet):
    """Normalize a Tweet (simplified) as opposed to:
    ------------------------------------------------
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
    -----------------------------------------------

    Args:
        tweet (str): Tweet to normalize

    Returns:
        str: normalized Tweet
    """
    tokens = TweetTokenizer().tokenize(tweet.replace("’", "'"))
    normTweet = " ".join([normalize_token_simplified(token) for token in tokens])

    normTweet = (
        normTweet.replace("n 't", "n't")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("p . m .", "pm")
            .replace("p . m", "pm")
            .replace("a . m .", "am")
            .replace("a . m", "am")
    )
    return " ".join(normTweet.split())

def get_response(request_params, prompt, context):
    request_params['prompt'] = prompt
    request_params['context'] = context

    response = requests.post(URI, json=request_params)

    if response.status_code == 200:
        result = response.json()['results'][0]['text']
        #print(prompt + result)
        return result
    else:
    	print(response)

def get_base_request_params(max_new_tokens = 200, stopping_strings = []):
    return {
        'prompt': None,
        'context': None,
        'max_new_tokens': 200,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'repetition_penalty': 1.2,
        'encoder_repetition_penalty': 1.0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        #'add_bos_token': True,
        #'truncation_length': 2048,
        #'ban_eos_token': False,
        #'skip_special_tokens': True,
        'stopping_strings': stopping_strings
    }

def generate_binary_balanced_dfs(all_labels, df, n = 65):
    """generates a list of balanced dataframes that have equal amount of positive and negative tweets for each label provided as parameter

    Args:
        all_labels (list<str>): List of labels
        df (pd.DataFrame): dataframe to sample samples from
        n (int, optional): Amount of positive and negative tweets respectively (balanced). Defaults to 65.

    Returns:
        list<pd.DataFrame>: List of balanced dataframes per label provided.
    """
    balanced_dfs = []
    for label in all_labels:
        # Initialize an empty DataFrame for the balanced dataset
        balanced_df = pd.DataFrame()
        # Get the rows with the current label
        label_rows = df[df['annotations'].apply(lambda x: label in x)]
        
        # Get the rows without the current label
        non_label_rows = df[df['annotations'].apply(lambda x: label not in x)]
        
        # Sample 65 rows with the current label
        sample_label_rows = label_rows.sample(n=n, random_state=42)
        
        # Sample 65 rows without the current label
        sample_non_label_rows = non_label_rows.sample(n=n, random_state=42)
        
        # Combine the samples
        combined_sample = pd.concat([sample_label_rows, sample_non_label_rows], ignore_index=True)
        
        # Add the samples to the balanced DataFrame
        balanced_df = pd.concat([balanced_df, combined_sample], ignore_index=True)

        balanced_dfs.append(balanced_df)

    return balanced_dfs

def generate_unlabeled_dataset():
    data_dir = "../data/unlabeled_data/"
    file_names = ["GRU_202012_tweets.csv", "IRA_202012_tweets.csv", "REA_0621_tweets.csv", "uganda_0621_tweets.csv", "venezuela_201901_2_tweets.csv"]
    
    dfs = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)

def get_response_wip(prompt, first_model_type, second_model_type = "", follow_up = "", prompting_type = "simple", context = "", openai_model = "", max_tokens = 200, temperature = 0.7, stop = None):
    
    valid_models = ["llama", "vicuna", "openassistant", "openai-davinci", "openai-gpt-3.5-turbo"]
    assert first_model_type in valid_models, "First model type needs to be one of the following: " + ", ".join(valid_models)
    first_model = get_model_by_type(first_model_type)

    if prompting_type == "two-way":
        if second_model_type == "":
            second_model = get_model_by_type(first_model_type)
        else:
            assert second_model_type in valid_models, "Second model type needs to be one of the following: " + ", ".join(valid_models)
            assert follow_up != "", "Follow up needs to be specified for two_way prompting type"
            second_model = get_model_by_type(second_model_type)

        if openai_model != "":
            #print("first prompt: ", prompt)
            first_response = first_model(prompt, context = context, model = openai_model)
            if "gpt" in second_model_type:
                first_response = [prompt, {"role": "assistant", "content": first_response}]
            #time.sleep(2)
            #print("First response: ", first_response)
            second_response = second_model(follow_up, context = prompt + first_response, model = openai_model)
            return second_response
        
    if prompting_type == "simple":
        return first_model(prompt, model = openai_model, max_tokens = max_tokens, temperature = temperature, stop = stop)


def get_openai_response(prompt, context = [], model = "gpt-3.5-turbo", max_tokens = 200, temperature = 0.7, stop = None):
    # Use OpenAI's ChatCompletion API to get the chatbot's response

    if "gpt" in model:
        messages = []
        if context != [] and context != "":
            for c in context:
                messages.append(c)
        messages.append(prompt)
    else:
        if context != "" and context != []:
            prompt = context + prompt
        #print("Context: ", context)
        #print("Full prompt: ", prompt)
    if model == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
            messages=messages,   # The conversation history up to this point, as a list of dictionaries
            max_tokens=200,        # The maximum number of tokens (words or subwords) in the generated response
            stop=None,              # The stopping sequence for the generated response, if any (not used here)
            temperature=0.7,        # The "creativity" of the generated response (higher temperature = more creative)
        )

    elif model == "davinci":
        response = openai.Completion.create(
            model="text-davinci-003",  # The name of the OpenAI chatbot model to use
            prompt=prompt,   # The conversation history up to this point, as a list of dictionaries
            max_tokens=200,        # The maximum number of tokens (words or subwords) in the generated response
            stop=None,              # The stopping sequence for the generated response, if any (not used here)
            temperature=0.7,        # The "creativity" of the generated response (higher temperature = more creative)
        )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    # If no response with text is found, return the first response's content (which may be empty)
    return response.choices[0].message.content

def get_openassistant_llama_30b_4bit_without_context_only_classification_v01(tweet_text, label):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nTweet: {tweet_text}\nClass: "
    context = ''
    return prompt, context

def get_openassistant_llama_30b_4bit_without_context_only_classification_v02(tweet_text, label):
    prompt = f"Give the tweet a binary class based on if it's about {label} or not.\n\nTweet: {tweet_text}\nClass: "
    context = ''
    return prompt, context

def get_openassistant_llama_30b_4bit_without_context_only_classification_v03(tweet_text, label, request_params):
    prompt = f"Assign 1 if the tweet is about {label}. Assign 0 if it is not about {label}.\n\nTweet: {tweet_text}\nClass: "
    context = ''
    request_params["max_new_tokens"] = 10
    return prompt, context, request_params

def get_openassistant_llama_30b_4bit_with_rules_only_classification(tweet_text, label, rules, request_params):
    prompt = f"Based on rules, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nRules: {rules}\nTweet: {tweet_text}\nClass: "
    request_params["max_new_tokens"] = 10
    return prompt, "", request_params

def get_openassistant_llama_30b_4bit_few_shot_prompt_only_classification_1_pos_example(tweet_text, label, example_tweet):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 1\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openassistant_llama_30b_4bit_few_shot_prompt_only_classification_1_neg_example(tweet_text, label, example_tweet):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openassistant_llama_30b_4bit_few_shot_prompt_only_classification_1_random_example(tweet_text, label, example_tweet, example_tweet_label):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label}\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openassistant_llama_30b_4bit_few_shot_prompt_only_classification_1_pos_1_neg_example(tweet_text, label, pos_example_tweet, neg_example_tweet, request_params):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {pos_example_tweet}\nClass: 1\nExample Tweet: {neg_example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
    request_params["max_new_tokens"] = 10
    return prompt, '', request_params

def get_openassistant_llama_30b_4bit_few_shot_prompt_only_classification_n_random_example(tweet_text, label, example_tweets, request_params):
    example_tweets_str = ""
    for example_tweet in example_tweets:
        example_tweets_str += f"\nExample Tweet: {normalize_tweet_simplified(example_tweet[0])}\nClass: {example_tweet[1]}"
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.{example_tweets_str}\n\nTweet: {tweet_text}\nClass: "
    request_params["max_new_tokens"] = 10
    return prompt, "", request_params

def get_openassistant_llama_30b_4bit_without_context_elaboration_first(tweet_text, label):
    prompt = f"Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\nElaboration: "
    context = ''
    return prompt, context

def get_vicuna_prompt_with_rules_only_classification(tweet_text, label, rules, request_params):
    prompt = f"### Human: Based on rules, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nRules: {rules}\nTweet: {tweet_text}\n### Assistant:\nClass:"
    context = ''
    request_params["max_new_tokens"] = 10
    return prompt, context, request_params

def get_vicuna_prompt_with_rules_elaboration_first(tweet_text, label, rules, request_params):
    prompt = f"### Human: Based on rules, elaborate whether you think the Tweet is about {label}.\nRules: {rules}\nTweet: {tweet_text}\n### Assistant:\nElaboration: "
    context = ''
    request_params["max_new_tokens"] = 400
    return prompt, context, request_params

# This is inefficient because 2 calls are needed
def get_vicuna_prompt_without_context_elaboration_first(tweet_text, label):
    prompt = f"### Human: Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\n### Assistant:\nElaboration: "
    #the followup starts with ":" because vicuna hallucinates ### Human which is set as the stopping token. 
    # Since ### Human is the stopping token and due to the way the API works, a ":" is added so that the total prompt is the initial prompt with the ### Human token followed by ":", a new line and then the followup prompt.
    followup = f":\n Assign class 1 for {label} or 0 for not. \n###Assistant:\nClass: " 
    return prompt, followup

def get_vicuna_prompt_without_context_elaboration_first_v02(tweet_text, label):
    prompt = f"### Human: Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\n### Assistant:\nElaboration: "
    followup = f"### Human: Based on the elaboration, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nTweet: {tweet_text}\nElaboration: [ELABORATION]\n### Assistant:\nClass: "
    return prompt, followup

def get_vicuna_prompt_without_context_elaboration_first_v03(tweet_text, label):
    prompt = f"### Human: Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\n### Assistant:\nElaboration: "
    followup = f"### Human: Based on the elaboration, assign 1 if it's about {label} or 0 if not.\nElaboration: [ELABORATION]\n### Assistant:\nClass: "
    return prompt, followup

def get_vicuna_prompt_without_context_only_classification(tweet_text, label):
    prompt = f"### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\n### Assistant:\nClass: "
    context = ''
    return prompt, context

def get_vicuna_prompt_without_context_only_classification_v02(tweet_text, label):
    prompt = f"### Human: Give the tweet a binary class based on if it's about {label} or not.\n\nTweet: {tweet_text}\n### Assistant:\nClass: "
    context = ''
    return prompt, context

def get_vicuna_prompt_without_context_only_classification_v03(tweet_text, label):
    prompt = f"### Human: Assign 1 if the tweet is about {label}. Assign 0 if it is not about {label}.\n\nTweet: {tweet_text}\n### Assistant:\nClass: "
    context = ''
    return prompt, context
    
def get_vicuna_prompt_without_context_only_classification_v04(tweet_text, label):
    prompt = f"### Human: Is the Text about {label}? Answer with True or False.\n\nTweet: {tweet_text}\n### Assistant:\nResponse: "
    context = ''
    return prompt, context

def get_vicuna_few_shot_prompt_only_classification_1_pos_example(tweet_text, label, example_tweet):
    prompt = f"### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 1\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_vicuna_few_shot_prompt_only_classification_1_neg_example(tweet_text, label, example_tweet):
    prompt = f"### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_vicuna_few_shot_prompt_only_classification_1_random_example(tweet_text, label, example_tweet, example_tweet_label):
    prompt = f"### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label}\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_vicuna_few_shot_prompt_only_classification_n_random_example(tweet_text, label, example_tweets):
    example_tweets_str = ""
    for example_tweet in example_tweets:
        example_tweets_str += f"\nExample Tweet: {normalize_tweet_simplified(example_tweet[0])}\nClass: {example_tweet[1]}"
    prompt = f"### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.{example_tweets_str}\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_vicuna_few_shot_prompt_only_classification_n_random_example_v02(tweet_text, label, example_tweets):
    example_tweets_str = ""
    for example_tweet in example_tweets:
        example_tweets_str += f"\n\nTweet: {normalize_tweet_simplified(example_tweet[0])}\n### Assistant:\nClass: {example_tweet[1]}"
    prompt = f"### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.{example_tweets_str}\n\nTweet: {tweet_text}\n### Assistant:\nClass: "
    return prompt, ""

def get_vicuna_few_shot_prompt_only_classification_1_pos_1_neg_example(tweet_text, label, pos_example_tweet, neg_example_tweet):
    prompt = f"### Human: Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {pos_example_tweet}\nClass: 1\nExample Tweet: {neg_example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_gpt4xalpaca_4bit_prompt_basic_elaboration_first(tweet_text, label):
    prompt = f'### Instruction:\nElaborate on whether you think the Tweet is about {label} or something else.\n\nTweet: {tweet_text}\n\n### Response:'
    context = ''
    return prompt, context

def get_gpt4xalpaca_4bit_prompt_without_context_only_classification(tweet_text, label, request_params):
    instruction = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class."
    prompt = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{tweet_text}\n\n### Response:\n'
    context = ''
    request_params["max_new_tokens"] = 10
    return prompt, context, request_params

def get_gpt4xalpaca_4bit_prompt_without_context_only_classification_v02(tweet_text, label, request_params):
    instruction = f"Give the tweet a binary class based on if it's about {label} or not."
    prompt = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{tweet_text}\n\n### Response:\n'
    context = ''
    request_params["max_new_tokens"] = 10
    return prompt, context, request_params

def get_gpt4xalpaca_4bit_prompt_without_context_only_classification_v03(tweet_text, label, request_params):
    instruction = f"Assign 1 if the tweet is about {label}. Assign 0 if it is not about {label}."
    prompt = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{tweet_text}\n\n### Response:\n'
    context = ''
    request_params["max_new_tokens"] = 10
    return prompt, context, request_params

def get_gpt4xalpaca_4bit_prompt_without_context_elaboration_first_v04(tweet_text, label, request_params):
    instruction = f"Classify the Tweet based on if it's about {label}. Give an explanation using \"Explanation:\" then classify using \"Class:\" as 1 or 0."
    prompt = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{tweet_text}\n\n### Response:\n'
    context = ''
    request_params["max_new_tokens"] = 400
    return prompt, context, request_params

def get_vicuna_prompt_without_context_elaboration_first_v04(tweet_text, label, request_params):
    prompt = f"### Human:\nClassify the Tweet based on if it's about {label}. Give an explanation using \"Explanation:\" then classify using \"Class:\" as 1 or 0.\n\nTweet:\n{tweet_text}\n\n### Assistant:\n"
    context = ''
    request_params["max_new_tokens"] = 400
    return prompt, context, request_params

def get_openassistant_llama_30b_4bit_prompt_without_context_elaboration_first_v04(tweet_text, label, request_params):
    prompt = f"### Instruction:\nClassify the Tweet based on if it's about {label}. Give an explanation using \"Explanation:\" then classify using \"Class:\" as 1 or 0.\n\nTweet:\n{tweet_text}\n\n### Response:\n"
    context = ''
    request_params["max_new_tokens"] = 400
    return prompt, context, request_params

# TODO: maybe use the same "Class: " token for output as in vicuna
def get_openai_prompt_with_rules_elaboration_first(tweet_text, label, rules):
    prompt = f"Based on rules, elaborate whether you think the Tweet is about {label}.\nRules: {rules}\nTweet: {tweet_text}\nElaborations: "
    followup = f"\nAssign the label 1 if it's about {label} or 0 for not based on the elaboration. Only output the number."
    return prompt, followup

def get_openai_prompt_without_context_elaboration_first(tweet_text, label):
    prompt = f"Elaborate on whether you think the Tweet is about {label} or something else.\n\nTweet: {tweet_text}\n\n"
    followup = f"\nAssign the label 1 if it's about {label} or 0 for not based on the elaboration. Only output the number."
    return prompt, followup

# last TODO: Here it's correct check if its correct 
def get_openai_prompt_without_context_elaboration_first_v02(tweet_text, label):
    prompt = f"Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\nElaboration: "
    followup = f"Based on the elaboration, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nTweet: {tweet_text}\nElaboration: [ELABORATION]\nClass: "
    return prompt, followup

def get_openai_prompt_without_context_elaboration_first_v03(tweet_text, label):
    prompt = f"Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\nElaboration: "
    followup = f"Based on the elaboration, assign 1 if it's about {label} or 0 if not.\nElaboration: [ELABORATION]\nClass: "
    return prompt, followup

def get_openai_prompt_without_context_only_classification(tweet_text, label):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_prompt_without_context_only_classification_v02(tweet_text, label):
    prompt = f"Give the tweet a binary class based on if it's about {label} or not.\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_prompt_without_context_only_classification_v03(tweet_text, label):
    prompt = f"Assign 1 if the tweet is about {label}. Assign 0 if it is not about {label}.\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_prompt_with_rules_only_classification(tweet_text, label, rules):
    prompt = f"Based on rules, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nRules: {rules}\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_1_pos_example(tweet_text, label, example_tweet):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 1\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_1_neg_example(tweet_text, label, example_tweet):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_1_random_example(tweet_text, label, example_tweet, example_tweet_label):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label}\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_3_random_example(tweet_text, label, example_tweet, example_tweet_label1, example_tweet2, example_tweet_label2, example_tweet3, example_tweet_label3):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_1_pos_1_neg_example(tweet_text, label, pos_example_tweet, neg_example_tweet):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {pos_example_tweet}\nClass: 1\nExample Tweet: {neg_example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_positive_example(df, label, exclude_tweet):
    pos_example_df = df[(df['annotations'].apply(lambda x: label in x)) & (df['text'] != exclude_tweet)]
    pos_example_tweet = pos_example_df.sample(n=1, random_state=42)['text'].values[0]
    return pos_example_tweet

def get_negative_example(df, label, exclude_tweet):
    neg_example_df = df[(df['annotations'].apply(lambda x: label not in x)) & (df['text'] != exclude_tweet)]
    neg_example_tweet = neg_example_df.sample(n=1, random_state=42)['text'].values[0]
    return neg_example_tweet

def get_random_examples(df, label, exclude_tweet, n, random_state = 42):
    """_summary_

    Args:
        df (pd.dataframe): df to retrieve random examples
        label (str): label to check random examples with, if set to "multilabel", it will return the true label in string format of the random tweets
        exclude_tweet (str): text of tweet to be excluded from returned list
        n (int): amount of random tweets retrieved

    Returns:
        list(tuple): returns a list of tuples of format (tweet_text, 1/0 depending on whether it pertains to the label) 
    """
    # Exclude the specific tweet
    df = df[df['text'] != exclude_tweet]
    
    # Sample n random examples
    sampled_df = df.sample(n=n, random_state=random_state)

    values = sampled_df['text'].values
    if label == "multilabel":
        labels = sampled_df['annotations'].values
    else:
        labels = sampled_df['annotations'].apply(lambda x: int(label in x)).values
    
    # Return a list of tuples, each containing the tweet text and its annotations
    return list(zip(values, labels))

# TODO: maybe use the same "Class: " token for output as in vicuna
def get_openai_prompt_with_rules_elaboration_first(tweet_text, label, rules):
    prompt = f"Based on rules, elaborate whether you think the Tweet is about {label}.\nRules: {rules}\nTweet: {tweet_text}\nElaborations: "
    followup = f"\nAssign the label 1 if it's about {label} or 0 for not based on the elaboration. Only output the number."
    return prompt, followup

def get_openai_prompt_without_context_elaboration_first(tweet_text, label):
    prompt = f"Elaborate on whether you think the Tweet is about {label} or something else.\n\nTweet: {tweet_text}\n\n"
    followup = f"\nAssign the label 1 if it's about {label} or 0 for not based on the elaboration. Only output the number."
    return prompt, followup

# last TODO: Here it's correct check if its correct 
def get_openai_prompt_without_context_elaboration_first_v02(tweet_text, label):
    prompt = f"Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\nElaboration: "
    followup = f"Based on the elaboration, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nTweet: {tweet_text}\nElaboration: [ELABORATION]\nClass: "
    return prompt, followup

def get_openai_prompt_without_context_elaboration_first_v03(tweet_text, label):
    prompt = f"Elaborate on whether you think the Tweet is about {label} or something else.\nTweet: {tweet_text}\nElaboration: "
    followup = f"Based on the elaboration, assign 1 if it's about {label} or 0 if not.\nElaboration: [ELABORATION]\nClass: "
    return prompt, followup

def get_openai_prompt_without_context_elaboration_first_v04(tweet_text, label, request_params):
    prompt = f"Classify the Tweet based on if it's about {label}. Give an explanation using \"Explanation:\" then classify using \"Class:\" as 1 or 0.\n\nTweet:\n{tweet_text}\n\nExplanation:"
    context = ''
    request_params["max_new_tokens"] = 400
    return prompt, context, request_params

def get_openai_prompt_without_context_only_classification(tweet_text, label):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_prompt_without_context_only_classification_v02(tweet_text, label):
    prompt = f"Give the tweet a binary class based on if it's about {label} or not.\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_prompt_without_context_only_classification_v03(tweet_text, label):
    prompt = f"Assign 1 if the tweet is about {label}. Assign 0 if it is not about {label}.\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_prompt_with_rules_only_classification(tweet_text, label, rules):
    prompt = f"Based on rules, classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nRules: {rules}\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_1_pos_example(tweet_text, label, example_tweet):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 1\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_1_neg_example(tweet_text, label, example_tweet):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_1_random_example(tweet_text, label, example_tweet, example_tweet_label):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label}\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_3_random_example(tweet_text, label, example_tweet, example_tweet_label1, example_tweet2, example_tweet_label2, example_tweet3, example_tweet_label3):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {example_tweet}\nClass: {example_tweet_label1}\nExample Tweet: {example_tweet2}\nClass: {example_tweet_label2}\nExample Tweet: {example_tweet3}\nClass: {example_tweet_label3}\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_openai_few_shot_prompt_only_classification_1_pos_1_neg_example(tweet_text, label, pos_example_tweet, neg_example_tweet):
    prompt = f"Classify the Tweet based on if it's about {label}. Use 1 or 0 as class.\nExample Tweet: {pos_example_tweet}\nClass: 1\nExample Tweet: {neg_example_tweet}\nClass: 0\n\nTweet: {tweet_text}\nClass: "
    return prompt, ""

def get_positive_example(df, label, exclude_tweet):
    pos_example_df = df[(df['annotations'].apply(lambda x: label in x)) & (df['text'] != exclude_tweet)]
    pos_example_tweet = pos_example_df.sample(n=1, random_state=42)['text'].values[0]
    return pos_example_tweet

def get_negative_example(df, label, exclude_tweet):
    neg_example_df = df[(df['annotations'].apply(lambda x: label not in x)) & (df['text'] != exclude_tweet)]
    neg_example_tweet = neg_example_df.sample(n=1, random_state=42)['text'].values[0]
    return neg_example_tweet

def get_random_examples(df, label, exclude_tweet, n):
    # Exclude the specific tweet
    df = df[df['text'] != exclude_tweet]
    
    # Sample n random examples
    sampled_df = df.sample(n=n, random_state=42)
    
    # Return a list of tuples, each containing the tweet text and its annotations
    return list(zip(sampled_df['text'].values, sampled_df['annotations'].apply(lambda x: int(label in x)).values))

def get_model_by_type(model_type):
    if model_type == "llama":
        return #get_llama_response
    elif model_type == "vicuna":
        return #get_vicuna_response
    elif model_type == "openassistant":
        return #get_openassistant_response
    elif "openai" in model_type:
        return get_openai_response
    elif "gpt-3.5" in model_type:
        return get_openai_response