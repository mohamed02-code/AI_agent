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
import math

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
def as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"true", "1", "yes"}
    return bool(x)

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


def plot_df_pie(results_df):
    """
    Plots per-model pie charts of Correct / Wrong / Invalid outcomes.
    Matches the style of the earlier correctness pie chart.
    """
    # Get unique models
    model_ids = results_df["model_id"].unique()
    n_models = len(model_ids)

    ncols = 2
    nrows = (n_models + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    categories = ["correct", "wrong", "invalid"]
    colors = ["mediumseagreen", "tomato", "slategray"]

    for i, model_id in enumerate(model_ids):
        model_df = results_df[results_df["model_id"] == model_id]

        # Enforce consistent category order
        counts = model_df["type"].value_counts()
        counts = counts.reindex(categories, fill_value=0)

        ax = axes[i]
        ax.pie(
            counts,
            labels=[c.capitalize() for c in counts.index],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "black"}
        )
        ax.set_title(f"Model: {model_id}")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

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

def plot_recomputed_model_pies(df):
    # Remove the _ALL_ row
    df = df[df["model_id"] != "__ALL__"]

    num_models = len(df)
    cols = 2
    rows = math.ceil(num_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, df.iterrows()):
        labels = ['Correct', 'Incorrect', 'Invalid']
        values = [row['Correct'], row['Incorrect'], row['Invalid']]
        ax.pie(values, labels=labels, autopct='%1.1f%%',
               colors=["mediumseagreen", "tomato", "slategray"], startangle=90)
        ax.set_title(f"Model: {row['model_id']}")

    # Hide any unused subplots if df length is odd
    for ax in axes[len(df):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
def plot_correctness_calculated_pie(results_df, model_ids=None):
    """
    Plots pie charts of Correct / Incorrect / Invalid answers
    for each model_id in the DataFrame, with consistent colors.
    """
    df = results_df.copy()

    # normalize labels to lower-case
    df["status_recomputed"] = df["status_recomputed"].str.lower()

    # filter by given model_ids
    if model_ids is not None:
        df = df[df["model_id"].isin(model_ids)]
        if df.empty:
            print(f"No results for model_ids {model_ids}")
            return
        model_list = model_ids
    else:
        model_list = df["model_id"].unique()

    # define label-color mapping
    categories = ["correct", "incorrect", "invalid"]
    colors = ["mediumseagreen", "tomato", "slategray"]

    n_models = len(model_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]  # make iterable if only one model

    for ax, model_id in zip(axes, model_list):
        model_df = df[df["model_id"] == model_id]
        counts = model_df["status_recomputed"].value_counts()

        # reindex to enforce all categories present
        counts = counts.reindex(categories, fill_value=0)

        if counts.sum() == 0:
            # nothing to plot, show placeholder text
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.axis("off")
        else:
            ax.pie(
                counts,
                labels=counts.index,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
                wedgeprops={"edgecolor": "black"}
            )
        ax.set_title(f"Correctness — {model_id}")

    plt.tight_layout()
    plt.show()