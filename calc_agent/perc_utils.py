from datasets import load_dataset
import pandas as pd
import ast
import json
import matplotlib.pyplot as plt
import os
import requests
import sys
import re
import numpy as np
import asyncio
import aiohttp
from typing import List
import seaborn as sns
import builtins 
from jinja2 import Environment, DictLoader
from typing import List, Tuple
from collections import defaultdict

# Load dataset
dataset = load_dataset("ncbi/MedCalc-Bench-v1.0")
df = dataset["train"].to_pandas()  # or "test"
df_test = dataset["test"].to_pandas()  # or "test"

perc_df = df[df["Calculator Name"] == "PERC Rule for Pulmonary Embolism"]

perc_df_test = df_test[df_test["Calculator Name"] == "PERC Rule for Pulmonary Embolism"]


#Testing calcualtor
def perc_score(
    age,
    heart_rate,
    oxygen_saturation,
    has_hemoptysis=False,
    on_estrogen=False,
    history_dvt_pe=False,
    unilateral_leg_swelling=False,
    recent_trauma_or_surgery=False
):
    """
    Calculate the PERC rule score.
    Returns the number of failed criteria (0 = all passed, 8 = all failed).
    """
    if age is None or heart_rate is None or oxygen_saturation is None:
        return None  # Signal that calculation can't be done
    score = 0
    if age >= 50:
        score += 1
    if heart_rate >= 100:
        score += 1
    if oxygen_saturation < 95:
        score += 1
    if has_hemoptysis:
        score += 1
    if on_estrogen:
        score += 1
    if history_dvt_pe:
        score += 1
    if unilateral_leg_swelling:
        score += 1
    if recent_trauma_or_surgery:
        score += 1
    return score


# %% [markdown]
# ### Function to safely parse the "Relevant Entities" column

# %%

''''This function safely parses a string representation (val) of a dictionary or list similar to to_dict()'''
def safe_parse(val):
    if isinstance(val, dict):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None

# Using perc_df["Relevant Entities"].to_dict() to parse the "Relevant Entities" column produces the same result as perc_df["Parsed Entities"].
perc_df["Parsed Entities"] = perc_df["Relevant Entities"].apply(safe_parse)


entities_df = pd.json_normalize(perc_df["Parsed Entities"])


def get_value(item):
    if isinstance(item, list) and item:
        return item[0]
    return item

entities_cleaned = entities_df.map(get_value)

'''This function extracts the first element from a list or returns the item itself if it's not a list.
Mainly to account for values with units such as ["73","years]'''
def extract_number(val):
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    elif isinstance(val, (int, float)):
        return val
    else:
        return None
    
def calculate_perc_score_from_entities(entities):
    """Extracts values from entity dict and returns the computed PERC score."""
    age = extract_number(get_value(entities.get("age")))
    heart_rate = extract_number(get_value(entities.get("Heart Rate or Pulse")))
    o2_sat = extract_number(get_value(entities.get("O₂ saturation percentage")))

    hemoptysis = get_value(entities.get("Hemoptysis"), False)
    hormone_use = get_value(entities.get("Hormone use"), False)
    prior_pe_dvt = get_value(entities.get("Previously documented Deep Vein Thrombosis"), False) or \
                   get_value(entities.get("Previously Documented Pulmonary Embolism"), False)
    leg_swelling = get_value(entities.get("Unilateral Leg Swelling"), False)
    recent_surgery_trauma = get_value(entities.get("Recent surgery or trauma"), False)

    return perc_score(age, heart_rate, o2_sat, hemoptysis, hormone_use, prior_pe_dvt, leg_swelling, recent_surgery_trauma)



# %%
# Available models 
model_ids = [
    "Llama-3.3-70B-Instruct",#
    "Llama3-Med42-70B",#
    "Llama3-Med42-70B-32k",
    "Llama3-Med42-70B-DGX",
    "Llama3-Med42-DPO-70B",
    "Med42-Qwen2.5-72B-v3-bi",#
    "Med42-R1-Qwen3-4B",#
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-70b-Instruct",
    # "gliner-gte-small",
    # "gliner-multitask-large-v0.5",
    # "gte-small",
    # "thenlper/gte-small",
    # "whisper-large-v3-turbo",
    # "whisper-large-v3-turbo-vllm"
]
#Models that I will be using
model_ids = [
    "Llama-3.3-70B-Instruct",
    "Med42-Qwen2.5-72B-v3-bi"
]


# %% [markdown]
# ### Function to run models with different prompts among other params and plot the data


def extract_json_from_text(text: str):
    '''Extracts a JSON object from a text string. Assumes only one JSON object is present in the text.'''
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        parsed = json.loads(json_str)
        
        # Ensure "Answer" field is an integer if present
        if "Answer" in parsed:
            try:
                parsed["Answer"] = int(float(parsed["Answer"]))
            except (ValueError, TypeError):
                # If conversion fails, you could set to None or raise an error
                parsed["Answer"] = None
        
        return parsed
    except (ValueError, json.JSONDecodeError):
        return None

    


'''Attempts to extract a PERC score from a text string.
Returns a dictionary with "Answer" and optional "Note" keys.'''
def extract_score_from_text(text):
    if not isinstance(text, str):
        return None

    text = text.strip().lower()

    # Early rejection: misunderstood task
    rejection_keywords = [
        "not applicable", "cannot calculate", "not appropriate", "not designed",
        "does not apply", "score is not valid", "wrong score",
        "perc is not for", "perc (percutaneous|pneumonia|coronary|emergency)"
    ]
    if any(re.search(kw, text) for kw in rejection_keywords):
        return {"Answer": None, "Note": "LLM misunderstood or refused to calculate PERC score"}

    # Ensure correct PERC context
    if "pulmonary embolism" not in text and "perc criteria" in text:
        return {"Answer": None, "Note": "PERC criteria mentioned but not in PE context"}

    # Match known phrasing
    patterns = [
        r'\bperc\s*score\s*(?:is|=|of|:)?\s*(\d{1,2})\b',
        r'\btotal\s*score\s*(?:is|=|:)?\s*(\d{1,2})\b',
        r'\bscore\s*(?:is|=|:)?\s*(\d{1,2})\b',
        r'\banswer\s*(?:is|=|:)?\s*(\d{1,2})\b',
        r'\bmeets\s*(\d{1,2})\s+criteria',
        r'\b(\d{1,2})\s+criteria\s+met',
        r'\b(\d{1,2})\s+out\s+of\s+8\s+criteria',
        r'\b(\d{1,2})\s*/\s*8\s+criteria'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return {"Answer": int(match.group(1))}
            except ValueError:
                continue

    # Fallback 1: binary checklist count
    checklist_matches = re.findall(r':\s*([01])\b', text)
    if checklist_matches and len(checklist_matches) <= 8:
        try:
            score = sum(int(x) for x in checklist_matches)
            return {"Answer": score}
        except Exception:
            pass

    # Fallback 2: checklist-like marks
    checkbox_like = re.findall(r'-\s*\[(x|1)\]', text, re.IGNORECASE)
    if checkbox_like:
        return {"Answer": len(checkbox_like)}

    # Fallback 3: keyword-based approximate tally
    yes_count = len(re.findall(r'\b(yes|1|true|present)\b', text, re.IGNORECASE))
    no_count = len(re.findall(r'\b(no|0|false|absent)\b', text, re.IGNORECASE))
    if 0 < yes_count <= 8 and yes_count + no_count >= 5:
        return {"Answer": yes_count, "Note": "Heuristically inferred from 'yes' counts"}

    return {"Answer": None, "Note": "No identifiable score"}

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





# %%

# Configuration
BASE_URL = "https://hive.g42healthcare.ai/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
CONCURRENT_REQUESTS = 10  # Limit the number of parallel requests

# Async function to get LLM response
'''Async function to get chat completion from the LLM API. Takes in a session, semaphore, model ID, system instruction, and user instruction.
Returns the content of the first choice in the response.'''
async def get_chat_completion(session, sem, model: str, system_instruction: str, user_instruction: str,
                              temperature,max_token) -> str:
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction}
        ],
        "temperature": temperature,
        "stream": False,
        "max_tokens": max_token
    }
    
    async with sem:
        async with session.post(BASE_URL, headers=HEADERS, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["message"]["content"]

''' A function that runs a batch of prompts concurrently using asyncio and aiohttp.
It takes a model ID, system instruction, and a list of prompts, and returns the results.
It uses a semaphore to limit the number of concurrent requests.'''
async def run_batch_old(model: str, system_instruction: str, prompts: List[str],temperature,max_token) -> List[str]:
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_chat_completion(session, sem, model, system_instruction, prompt,temperature,max_token)
            for prompt in prompts
        ]

        results = []
        total = len(tasks)
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            progress = (completed / total) * 100
            print(f"\rBatch Progress: {progress:.1f}% ({completed}/{total})", end="", flush=True)

        print()  # newline after completion
        return results

# Option 2: Return list of dictionaries with prompt and result
async def run_batch(model: str, system_instruction: str, prompts: List[str], 
                           temperature, max_token):
    """Returns a list of dictionaries, each containing 'prompt' and 'result' keys."""
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_chat_completion(session, sem, model, system_instruction, prompt, temperature, max_token)
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Create list of dictionaries with prompt and result
        prompt_result_list = []
        for i, result in enumerate(results):
            prompt_result_list.append({
                "prompt": prompts[i],
                "result": result
            })
            progress = ((i + 1) / len(prompts)) * 100
            print(f"\rBatch Progress: {progress:.1f}% ({i + 1}/{len(prompts)})", end="", flush=True)
        print()
        return prompt_result_list
 
template_str = """
{% if include_relevant_entities %}
Relevant Entities:
{{ entities }}

{% endif %}
Patient Note:
{{ note }}

{{ prompt }}
"""

env = Environment(loader=DictLoader({'perc_prompt': template_str}))
template = env.get_template('perc_prompt')
# %%
''' A function that runs models with output.
It takes a system instruction, a list of model IDs, and optional parameters for including relevant entities,
sampling, and full DataFrame usage.
It returns invalid outputs, wrong outputs, a summary DataFrame, and parsed results per model.'''
async def run_models_with_output(
    sys_instruct,
    model_ids: List[str],
    include_relevant_entities=True,
    sample=10,
    full_df=False,
    prompt="Calculate PERC score for this patient.",
    temperature=0.0,
    max_tokens=1000
):
    results_data = []
    subset_df = perc_df.head(sample).reset_index(drop=True)
    iterate_df = perc_df.reset_index(drop=True) if full_df else subset_df

    for model_id in model_ids:
        print(f"\n=== Evaluating model: {model_id} ===\n")
        prompts = []

        for idx, row in iterate_df.iterrows():
            entities = ast.literal_eval(row["Relevant Entities"]) if include_relevant_entities else None
            user_instruction = template.render(
                note=row["Patient Note"],
                entities=row["Parsed Entities"],
                include_relevant_entities=include_relevant_entities,
                prompt=prompt
            )
            prompts.append(user_instruction)

        results_async = await run_batch(
            model=model_id,
            system_instruction=sys_instruct,
            prompts=prompts,
            temperature=temperature,
            max_token=max_tokens
        )

        for idx, row in iterate_df.iterrows():
            prompt_result = results_async[idx]
            reply = prompt_result["result"]
            entities = row["Parsed Entities"]
            ground_truth = row["Ground Truth Answer"]

            parsed = extract_json_from_text(reply)
            entry_type = "invalid"
            criteria = None
            predicted_score = None
            valid = False
            correct_flag = False

            if parsed is not None and "Answer" in parsed:
                try:
                    parsed["Answer"] = int(float(parsed["Answer"]))
                except Exception as e:
                    print(f"Failed to convert Answer to int: {parsed['Answer']}, error: {e}")
                    parsed["Answer"] = None

                answer = parsed.get("Answer")
                if answer is not None:
                    predicted_score = answer
                    criteria = {k: v for k, v in parsed.items() if k != "Answer"}
                    valid = True
                    if predicted_score == int(ground_truth):
                        entry_type = "correct"
                        correct_flag = True
                    else:
                        entry_type = "wrong"

            results_data.append({
                "model_id": model_id,
                "note": row["Patient Note"],
                "entities": entities,
                "ground_truth": ground_truth,
                "predicted": predicted_score,
                "parsed_criteria": criteria,
                "reply": reply,
                "type": entry_type,
                "prompt": prompt_result["prompt"],
                "valid": valid,
                "correct": correct_flag
            })

        print("\nDone ✅")

    results_df = pd.DataFrame(results_data)

    # Optional summary print
    print("\n=== Summary Counts ===")
    print(results_df["type"].value_counts())

    return results_df



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

def compute_truth_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare predicted PERC criteria to ground truth derived from parsed patient entities.
    Returns a DataFrame showing criterion-level accuracy per model.
    """
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ['true', 'yes', '1']
        return bool(val)

    rows = []

    for _, row in results_df.iterrows():
        model_id = row["model_id"]
        criteria_pred = None

        try:
            criteria_pred = row["parsed_criteria"]["Criteria"]
        except (TypeError, KeyError):
            continue  # skip invalid rows

        entities = row["entities"]

        # Extract numbers safely
        age = extract_number(entities.get("age"))
        hr = extract_number(entities.get("Heart Rate or Pulse"))
        o2 = extract_number(entities.get("O₂ saturation percentage"))

        # Construct truth based on entities
        truth = {
            "Age < 50": age is not None and age < 50,
            "HR < 100": hr is not None and hr < 100,
            "O₂ ≥ 95%": o2 is not None and o2 >= 95,
            "No hemoptysis": not to_bool(entities.get("Hemoptysis", False)),
            "No Hormone use": not to_bool(entities.get("Hormone use", False)),
            "No prior VTE or DVT": not (
                to_bool(entities.get("Previously documented Deep Vein Thrombosis", False)) or
                to_bool(entities.get("Previously Documented Pulmonary Embolism", False))
            ),
            "No unilateral leg swelling": not to_bool(entities.get("Unilateral Leg Swelling", False)),
            "No recent trauma or surgery": not to_bool(entities.get("Recent surgery or trauma", False)),
        }

        for criterion, true_val in truth.items():
            pred_val = criteria_pred.get(criterion)
            if pred_val is None:
                continue  # Missing prediction
            match = (to_bool(pred_val) == true_val)
            rows.append({
                "Model": model_id,
                "Criterion": criterion,
                "Correct": match
            })

    df = pd.DataFrame(rows)

    accuracy_df = (
        df.groupby(["Model", "Criterion"])["Correct"]
        .mean()
        .reset_index()
        .pivot(index="Criterion", columns="Model", values="Correct")
        .round(3)
    )

    return accuracy_df


# %%
'''A function that plots confusion matrices for each model and criterion based on parsed JSON outputs and patient data.
It creates a grid of subplots with each model's confusion matrix for each criterion.'''
def plot_confusion_matrices_by_model(parsed_json, patient_df):
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ['true', 'yes', '1']
        return bool(val)

    data = []

    for model_id, outputs in parsed_json.items():
        for i, output in enumerate(outputs):
            try:
                criteria_pred = output["parsed_criteria"]["Criteria"]
            except (TypeError, KeyError):
                continue

            row = patient_df.iloc[i]
            entities = row["Parsed Entities"]

            age = extract_number(entities.get("age"))
            hr = extract_number(entities.get("Heart Rate or Pulse"))
            o2 = extract_number(entities.get("O₂ saturation percentage"))

            truth = {
                "Age < 50": age is not None and age < 50,
                "HR < 100": hr is not None and hr < 100,
                "O₂ ≥ 95%": o2 is not None and o2 >= 95,
                "No hemoptysis": not to_bool(entities.get("Hemoptysis", False)),
                "No estrogen use": not to_bool(entities.get("Estrogen use", False)),
                "No prior VTE": not to_bool(entities.get("Prior VTE", False)),
                "No unilateral leg swelling": not to_bool(entities.get("Unilateral leg swelling", False)),
            }

            for criterion, true_val in truth.items():
                pred_val = criteria_pred.get(criterion)
                if pred_val is None:
                    continue
                data.append({
                    "Model": model_id,
                    "Criterion": criterion,
                    "True": true_val,
                    "Pred": pred_val,
                })

    df = pd.DataFrame(data)

    models = df["Model"].unique()
    criteria = df["Criterion"].unique()

    nrows = len(models)
    ncols = len(criteria)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for i, model in enumerate(models):
        for j, criterion in enumerate(criteria):
            ax = axes[i][j]
            sub = df[(df["Model"] == model) & (df["Criterion"] == criterion)]
            cm = pd.crosstab(sub["True"], sub["Pred"])
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax)
            ax.set_title(f"{model} — {criterion}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


''' A function that plots the accuracy of PERC criteria by outcome (correct or incorrect) for each model.
It creates a heatmap for each outcome type, showing the accuracy of each criterion per model.'''
def plot_criteria_accuracy_by_outcome(results_df: pd.DataFrame):
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ['true', 'yes', '1']
        return bool(val)

    def compute_accuracy_df(filter_correct=None):
        rows = []

        for _, row in results_df.iterrows():
            model_id = row["model_id"]
            is_correct = row.get("correct", None)

            # Filter based on correctness if specified
            if filter_correct is not None and is_correct != filter_correct:
                continue

            try:
                criteria_pred = row["parsed_criteria"]["Criteria"]
            except (TypeError, KeyError):
                continue  # skip invalid rows

            entities = row["entities"]

            # Extract numbers safely
            age = extract_number(entities.get("age"))
            hr = extract_number(entities.get("Heart Rate or Pulse"))
            o2 = extract_number(entities.get("O₂ saturation percentage"))

            # Construct truth
            truth = {
                "Age < 50": age is not None and age < 50,
                "HR < 100": hr is not None and hr < 100,
                "O₂ ≥ 95%": o2 is not None and o2 >= 95,
                "No hemoptysis": not to_bool(entities.get("Hemoptysis", False)),
                "No Hormone use": not to_bool(entities.get("Hormone use", False)),
                "No prior VTE or DVT": not (
                    to_bool(entities.get("Previously documented Deep Vein Thrombosis", False)) or
                    to_bool(entities.get("Previously Documented Pulmonary Embolism", False))
                ),
                "No unilateral leg swelling": not to_bool(entities.get("Unilateral Leg Swelling", False)),
                "No recent trauma or surgery": not to_bool(entities.get("Recent surgery or trauma", False)),
            }

            for criterion, true_val in truth.items():
                pred_val = criteria_pred.get(criterion)
                if pred_val is None:
                    continue  # Missing prediction
                match = (to_bool(pred_val) == true_val)
                rows.append({
                    "Model": model_id,
                    "Criterion": criterion,
                    "Correct": match
                })

        df = pd.DataFrame(rows)

        summary = (
            df.groupby(["Model", "Criterion"])["Correct"]
            .mean()
            .reset_index()
            .pivot(index="Criterion", columns="Model", values="Correct")
            .round(3)
        )

        return summary

    # Compute 3 condition-specific DataFrames
    correct_df = compute_accuracy_df(filter_correct=True)
    incorrect_df = compute_accuracy_df(filter_correct=False)
    all_df = compute_accuracy_df(filter_correct=None)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

    for df, ax, title in zip(
        [correct_df, incorrect_df, all_df],
        axes,
        ["When Final Answer is Correct", "When Final Answer is Incorrect", "All Predictions"]
    ):
        if df.empty:
            ax.set_visible(False)
            continue

        sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=True, ax=ax)
        ax.set_title(f"Criterion Accuracy — {title}")
        ax.set_xlabel("Model")
        ax.set_ylabel("PERC Criterion")

    plt.tight_layout()
    plt.show()

'''Takes a parsed_json object (output from each model), Takes a patient_df (your DataFrame with patient notes and entities), Extracts and compares predicted vs actual PERC criteria, 
And produces pie charts showing how often models: Got all criteria right, Got one or more criteria wrong, Separately for correct and incorrect final predictions.'''
import matplotlib.pyplot as plt

def plot_criteria_accuracy_pie(results_df: pd.DataFrame):
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ['true', 'yes', '1']
        return bool(val)

    # Define the "truth" function based on entities (same as in compute_truth_table)
    def get_truth(entities):
        age = extract_number(entities.get("age"))
        hr = extract_number(entities.get("Heart Rate or Pulse"))
        o2 = extract_number(entities.get("O₂ saturation percentage"))
        return {
            "Age < 50": age is not None and age < 50,
            "HR < 100": hr is not None and hr < 100,
            "O₂ ≥ 95%": o2 is not None and o2 >= 95,
            "No hemoptysis": not to_bool(entities.get("Hemoptysis", False)),
            "No Hormone use": not to_bool(entities.get("Hormone use", False)),
            "No prior VTE or DVT": not (
                to_bool(entities.get("Previously documented Deep Vein Thrombosis", False)) or
                to_bool(entities.get("Previously Documented Pulmonary Embolism", False))
            ),
            "No unilateral leg swelling": not to_bool(entities.get("Unilateral Leg Swelling", False)),
            "No recent trauma or surgery": not to_bool(entities.get("Recent surgery or trauma", False)),
        }

    # Aggregate stats per model and correctness group
    model_stats = {}

    for model_id in results_df["model_id"].unique():
        df_model = results_df[results_df["model_id"] == model_id]
        correct_all = 0
        correct_some = 0
        wrong_all = 0
        wrong_some = 0

        for _, row in df_model.iterrows():
            try:
                criteria_pred = row["parsed_criteria"]["Criteria"]
            except (TypeError, KeyError):
                continue

            truth = get_truth(row["entities"])
            match_flags = []

            for criterion, true_val in truth.items():
                pred_val = criteria_pred.get(criterion)
                if pred_val is None:
                    continue
                match_flags.append(to_bool(pred_val) == true_val)

            if not match_flags:
                continue  # No criteria matched

            if all(match_flags):
                if row.get("correct", False):
                    correct_all += 1
                else:
                    wrong_all += 1
            else:
                if row.get("correct", False):
                    correct_some += 1
                else:
                    wrong_some += 1

        model_stats[model_id] = {
            "correct": {
                "All Criteria Correct": correct_all,
                "One or More Incorrect": correct_some
            },
            "incorrect": {
                "All Criteria Correct": wrong_all,
                "One or More Incorrect": wrong_some
            }
        }

    # Plot pie charts
    num_models = len(model_stats)
    fig, axes = plt.subplots(2, num_models, figsize=(6 * num_models, 12))

    if num_models == 1:
        axes = axes.reshape(2, 1)

    for col, (model_id, counts) in enumerate(model_stats.items()):
        for row_idx, key in enumerate(["correct", "incorrect"]):
            subset = counts[key]
            total = sum(subset.values())
            ax = axes[row_idx][col]

            if total == 0:
                ax.set_title(f"{model_id}\n(No {key} answers)")
                ax.axis("off")
                continue

            sizes = [subset["All Criteria Correct"], subset["One or More Incorrect"]]
            labels = ["All Criteria Correct", "One or More Incorrect"]
            colors = ["mediumseagreen", "tomato"]

            ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
            ax.set_title(f"{model_id} - {key.capitalize()}")

    plt.suptitle("Criteria Accuracy Breakdown per Model", fontsize=18)
    plt.tight_layout()
    plt.show()



# def llm(text:str) -> str:
#     pass

# def parser(text:str) -> dict:
#     pass

# def tool(args: dict) -> int:
#     pass


# def agent(input_message):
#     '''  '''
#     text_output = llm(input_message)
#     json_dict = parser(text_output)
#     perc_score = tool(json_dict)

#     explanation = llm(f"{perc_score}")
#     return perc_score



# %%
def print_correct_with_wrong_criteria(parsed_json, patient_df):
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ['true', 'yes', '1']
        return bool(val)

    def get_truth(entities):
        age = extract_number(entities.get("age"))
        hr = extract_number(entities.get("Heart Rate or Pulse"))
        o2 = extract_number(entities.get("O₂ saturation percentage"))
        return {
            "Age < 50": age is not None and age < 50,
            "HR < 100": hr is not None and hr < 100,
            "O₂ ≥ 95%": o2 is not None and o2 >= 95,
            "No hemoptysis": not to_bool(entities.get("Hemoptysis", False)),
            "No estrogen use": not to_bool(entities.get("Estrogen use", False)),
            "No prior VTE": not to_bool(entities.get("Prior VTE", False)),
            "No unilateral leg swelling": not to_bool(entities.get("Unilateral leg swelling", False)),
        }

    count = 0

    for model_id, outputs in parsed_json.items():
        for i, output in enumerate(outputs):
            if not output.get("correct"):
                continue  # Only focus on correct predictions

            try:
                criteria_pred = output["parsed_criteria"]["Criteria"]
            except (TypeError, KeyError):
                continue

            row = patient_df.iloc[i]
            entities = row["Parsed Entities"]
            truth = get_truth(entities)

            wrong_criteria = []
            for criterion, true_val in truth.items():
                pred_val = criteria_pred.get(criterion)
                if pred_val is None:
                    continue
                if pred_val != true_val:
                    wrong_criteria.append(criterion)

            if wrong_criteria:
                count += 1
                print(f"\n--- Correct Answer with Wrong Criteria {count} ---")
                print(f"Model: {model_id}")
                print(f"Wrong Criteria: {wrong_criteria}")
                print(f"Expected (Ground Truth): {output.get('ground_truth')} | Predicted: {output.get('predicted')}")
                print(f"\nParsed Entities:\n{entities}")
                print(f"\nPatient Note:\n{row.get('Patient Note', 'N/A')}")
                print(f"\nReply:\n{output.get('raw_reply', 'N/A')}")

    if count == 0:
        print("✅ No correct predictions with incorrect criteria.")


async def validate_model_outputs_with_json(
    sys_instruct: str,
    results_df: pd.DataFrame,
    temperature=0.0,
    max_tokens=500
):
    """
    Rerun validation prompts grouped by model_id from the results_df.
    Each model gets its own batch call with its respective prompts.
    Returns results_df with 'corrected_reply' and 'corrected_json' columns.
    """
    # Group prompts and tracking indices by model
    prompts_by_model = defaultdict(list)
    indices_by_model = defaultdict(list)

    for idx, row in results_df.iterrows():
        model_id = row["model_id"]
        if not row["valid"]:
            prompts_by_model[model_id].append("Skip: original output was invalid.")
        else:
            prompt = (
                "You are reviewing a previously generated PERC score JSON output and explanation.\n"
                "Your task is to check if the explanation, criteria, and final score are consistent.\n"
                "If anything is wrong, correct it. Otherwise, return the same JSON unchanged.\n"
                "Return only a single corrected JSON object, nothing else.\n\n"
                
                f"Original Reply:\n{row['reply']}"
            )
            prompts_by_model[model_id].append(prompt)

        indices_by_model[model_id].append(idx)

    # Initialize output containers
    corrected_replies = [None] * len(results_df)
    corrected_jsons = [None] * len(results_df)

    # Run validation per model_id
    for model_id, prompts in prompts_by_model.items():
        batch_results = await run_batch(
            model=model_id,
            system_instruction=sys_instruct,
            prompts=prompts,
            temperature=temperature,
            max_token=max_tokens
        )

        indices = indices_by_model[model_id]
        for i, res in enumerate(batch_results):
            idx = indices[i]
            raw_reply = res.get("result")
            corrected_replies[idx] = raw_reply
            corrected_jsons[idx] = extract_json_from_text(raw_reply)  # will still handle None/invalid cases

    # Add results back into a new dataframe
    results_df = results_df.copy()
    results_df["corrected_reply"] = corrected_replies
    results_df["corrected_json"] = corrected_jsons

    return results_df

def plot_corrected_json_verification_pie(results_df):
    """
    For each model, plots two pie charts:
    - One showing the original prediction outcome (correct/wrong/invalid)
    - One showing second-pass LLM verification outcome

    Uses 'predicted', 'ground_truth', and 'corrected_json' columns from results_df.
    """

    model_ids = results_df["model_id"].unique()
    n_models = len(model_ids)

    # Create 2 rows per model: one for original, one for second-pass
    ncols = 2
    nrows = n_models * 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * n_models))
    axes = axes.flatten()

    for model_index, model_id in enumerate(model_ids):
        model_df = results_df[results_df["model_id"] == model_id]

        # --- Original Prediction Evaluation ---
        correct = 0
        wrong = 0
        invalid = 0

        for _, row in model_df.iterrows():
            predicted = row.get("predicted")
            ground_truth = row.get("ground_truth")

            if predicted is None:
                invalid += 1
                continue

            try:
                if int(predicted) == int(ground_truth):
                    correct += 1
                else:
                    wrong += 1
            except Exception:
                invalid += 1

        values_orig = [correct, wrong, invalid]
        labels = ["Correct", "Wrong", "Invalid"]
        colors = ["mediumseagreen", "tomato", "slategray"]

        if sum(values_orig) > 0:
            ax_orig = axes[model_index * 2]
            ax_orig.pie(values_orig, labels=labels, autopct="%1.1f%%",
                        colors=colors, startangle=140)
            ax_orig.set_title(f"Model: {model_id} (Original Prediction)")

        # --- Second-Pass Verification Evaluation ---
        correct = 0
        wrong = 0
        invalid = 0

        for _, row in model_df.iterrows():
            predicted = row["predicted"]
            corrected_json = row.get("corrected_json")

            if not corrected_json or not isinstance(corrected_json, dict):
                invalid += 1
                continue

            verified_answer = corrected_json.get("Answer")

            try:
                verified_answer = int(float(verified_answer))
            except Exception:
                invalid += 1
                continue

            if verified_answer == predicted:
                correct += 1
            else:
                wrong += 1

        values_corr = [correct, wrong, invalid]
        labels_corr = ["Verified Correct", "Verified Wrong", "Invalid / Missing"]

        if sum(values_corr) > 0:
            ax_corr = axes[model_index * 2 + 1]
            ax_corr.pie(values_corr, labels=labels_corr, autopct="%1.1f%%",
                        colors=colors, startangle=140)
            ax_corr.set_title(f"Model: {model_id} (Second-Pass Verification)")

    # Hide any unused subplots
    for j in range(n_models * 2, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Prediction vs Second-Pass Verification per Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()





def add_calc_param_to_results(results_df):

    # Make a copy to avoid mutating original df
    df = results_df.copy()

    def safe_extract_json(reply):
        try:
            return extract_json_from_text(reply)
        except Exception:
            return None

    df["calc_param"] = df["reply"].apply(safe_extract_json)

    return df

def add_calculated_score(results_df):
    df = results_df.copy()

    def compute_score(params):
        if not isinstance(params, dict):
            return None
        try:
            return perc_score(
                params.get("age"),
                params.get("heart_rate"),
                params.get("oxygen_saturation"),
                params.get("has_hemoptysis", False),
                params.get("on_estrogen", False),
                params.get("history_dvt_pe", False),
                params.get("unilateral_leg_swelling", False),
                params.get("recent_trauma_or_surgery", False)
            )
        except Exception:
            return None

    df["calculated_score"] = int(df["calc_param"].apply(compute_score))
    return df


