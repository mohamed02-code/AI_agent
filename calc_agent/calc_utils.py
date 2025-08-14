from datasets import load_dataset
import pandas as pd
import ast
import json
import matplotlib.pyplot as plt
import os
import requests
import sys
import regex as re  
import numpy as np
import asyncio
import aiohttp
from typing import List,Any,Tuple,Dict,Mapping,Optional
import seaborn as sns
import builtins 
from jinja2 import Environment, DictLoader
from collections import defaultdict


''''This function safely parses a string representation (val) of a dictionary or list similar to to_dict()'''
def safe_parse(val):
    if isinstance(val, dict):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None

# Using perc_df["Relevant Entities"].to_dict() to parse the "Relevant Entities" column produces the same result as perc_df["Parsed Entities"].

def to_float(val):
    """Robustly parse numbers from strings or lists like ['73','years'] -> 73.0."""
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, str):
        # keep digits, dot, minus
        m = re.search(r'[-+]?\d*\.?\d+', val.replace(',', ''))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None

def get_value(item):
    if isinstance(item, list) and item:
        return item[0]
    return item

'''This function extracts the first element from a list or returns the item itself if it's not a list.
Mainly to account for values with units such as ["73","years]'''
def extract_number(val):
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    elif isinstance(val, (int, float)):
        return val
    else:
        return None



# %% [markdown]
# ### Function to run models with different prompts among other params and plot the data

JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
CURLY_BLOCK = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)  # recursive-ish fallback

def extract_json_from_text(text: str):
    if not isinstance(text, str): return None
    m = JSON_BLOCK.search(text)
    s = m.group(1) if m else (CURLY_BLOCK.search(text).group(0) if CURLY_BLOCK.search(text) else None)
    if not s: return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    # normalize Answer
    if "Answer" in obj:
        try: obj["Answer"] = int(float(obj["Answer"]))
        except Exception: obj["Answer"] = None
    return obj 


'''Plots a bar chart comparing the counts of correct, wrong, and invalid answers for each model.'''
def plot_df(results_df):
    x = np.arange(len(results_df))  # label locations
    width = 0.25  # narrower bar width to fit 3 bars per group

    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width,     results_df["correct"], width, label='Correct', color='mediumseagreen')
    bars2 = ax.bar(x,             results_df["wrong"],   width, label='Wrong',   color='tomato')
    bars3 = ax.bar(x + width,     results_df["invalid"], width, label='Invalid', color='slategray')

    ax.set_title("Correct, Wrong, and Invalid Answer Count per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model_id"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


'''Plots a pie chart for each model showing the distribution of correct, wrong, and invalid answers.'''
def plot_df_pie(results_df):
    # Get unique models
    model_ids = results_df["model_id"].unique()
    n_models = len(model_ids)

    ncols = 2
    nrows = (n_models + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for i, model_id in enumerate(model_ids):
        model_df = results_df[results_df["model_id"] == model_id]

        counts = model_df["type"].value_counts()
        labels = []
        values = []
        colors = []

        for label, color in zip(["correct", "wrong", "invalid"],
                                ['mediumseagreen', 'tomato', 'slategray']):
            if label in counts:
                labels.append(label.capitalize())
                values.append(counts[label])
                colors.append(color)

        if not values:
            continue  # Skip model if no data

        ax = axes[i]
        ax.pie(values, labels=labels, autopct='%1.1f%%',
               colors=colors, startangle=140)
        ax.set_title(f"Model: {model_id}")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Prediction Outcome Breakdown per Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# %% [markdown]
# #### Troubleshooting wrong/invalid replies
def print_troubleshooting_outputs(results_df):
    """
    Prints detailed troubleshooting info for wrong and invalid responses 
    based on the 'type' field in the results DataFrame.
    
    Args:
        results_df (pd.DataFrame): DataFrame with model outputs. Must include:
            ['model_id', 'note', 'entities', 'ground_truth', 'predicted', 
             'parsed_criteria', 'prompt', 'reply', 'type', 'valid', 'correct']
    """
    wrong_df = results_df[results_df['type'] == 'wrong']
    invalid_df = results_df[results_df['type'] == 'invalid']

    print(f"\n===== INCORRECT OUTPUTS ({len(wrong_df)}) =====")
    for i, row in wrong_df.iterrows():
        print(f"\n--- Wrong Reply {i + 1} ---")
        print(f"Model: {row['model_id']}")
        print(f"Expected: {row['ground_truth']} | Predicted: {row['predicted']}")
        print(f"Patient Note:\n{row['note']}")
        print(f"Entities:\n{row['entities']}")
        print(f"Parsed Criteria:\n{row['parsed_criteria']}")
        print(f"Prompt:\n{row['prompt']}")
        print(f"Reply:\n{row['reply']}")

    print(f"\n===== INVALID OUTPUTS ({len(invalid_df)}) =====")
    for i, row in invalid_df.iterrows():
        print(f"\n--- Invalid Reply {i + 1} ---")
        print(f"Model: {row['model_id']}")
        print(f"Expected: {row['ground_truth']}")
        print(f"Patient Note:\n{row['note']}")
        print(f"Entities:\n{row['entities']}")
        print(f"Prompt:\n{row['prompt']}")
        print(f"Reply:\n{row['reply']}")

    # Just raw replies from invalid if needed
    invalid_replies = invalid_df['reply'].tolist()
    for i, reply in enumerate(invalid_replies):
        print(f"\n--- Invalid Reply (Raw Only) {i + 1} ---\n{reply}")

