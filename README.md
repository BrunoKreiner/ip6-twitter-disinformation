# Classifying Twitter Disinformation Campaigns with Deep Learning

[![Test Model Training](https://github.com/BrunoKreiner/ip6-twitter-disinformation/actions/workflows/test_workflow.yml/badge.svg)](https://github.com/BrunoKreiner/ip6-twitter-disinformation/actions/workflows/test_workflow.yml)

## Project Structure

```
├── .github  
│   ├── workflows                   - github workflows
├── data                            - data folder
│   ├── labeled_data                - labeled data
│   ├── weak_labeled_data           - weakly labeled data
├── models                          - contains all model files 
├── notebooks                       - contains jupyter notebooks 
├── reports                         - contains figures and texts and model training history
│   └── figures
└── src                             - contains scripts, helpers and utility functions
│   └── eda.py                      - contains easy data augmentation source code
│   └── llm_utils.py                - contains helper functions to evaluate llms
│   └── prompt_utils.py             - contains helper functions to setup experiments and prompt llms
│   └── reproduce_model.py          - contains code to reproduce the models from the existing report
│   └── stats_streamlit.py          - contains code to launch streamlit dashboard
└── README.md
└── requirements.txt
```
