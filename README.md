# Classifying Twitter Disinformation Campaigns with Deep Learning

[![Test Model Training](https://github.com/BrunoKreiner/ip6-twitter-disinformation/actions/workflows/test_workflow.yml/badge.svg)](https://github.com/BrunoKreiner/ip6-twitter-disinformation/actions/workflows/test_workflow.yml)

## Project Structure

```
├── .github  
│   ├── workflows                   - github workflows
├── data                            - data folder
│   ├── labeled_data                - labeled data 
├── models                          - contains all model files 
├── notebooks                       - contains jupyter notebooks 
│   ├── reproduce_tables.ipynb      - contains code to reproduce existing report tables
├── reports                         - contains figures and texts and model metrics 
│   └── figures
└── src                             - contains scripts, helpers and utility functions
│   └── reproduce_model.py          - contains code to reproduce the models from the existing report
└── README.md
└── requirements.txt
```