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
    "Llama-3.3-70B-Instruct",#
    "Llama3-Med42-70B",#
    "Med42-Qwen2.5-72B-v3-bi"
]


# %% [markdown]
# ### Function to run models with different prompts among other params and plot the data

# %%
'''extracts a JSON object from a text string. Assues only one JSON object is present in the text.'''
def extract_json_from_text(text: str):
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        return json.loads(json_str)
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
    num_models = len(results_df)
    ncols = 2
    nrows = (num_models + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for i, row in results_df.iterrows():
        # Prepare label, value, color triplets
        all_labels = ['Correct', 'Incorrect', 'Invalid']
        all_counts = [row['correct'], row['wrong'], row['invalid']]
        all_colors = ['mediumseagreen', 'tomato', 'slategray']

        # Filter out zero-counts
        filtered = [(label, count, color) for label, count, color in zip(all_labels, all_counts, all_colors) if count > 0]
        if not filtered:
            continue  # skip completely empty pie

        labels, counts, colors = zip(*filtered)

        ax = axes[i]
        ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax.set_title(f"Model: {row['model_id']}")

    # Hide unused subplots
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
async def get_chat_completion(session, sem, model: str, system_instruction: str, user_instruction: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction}
        ],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 2000
    }

    async with sem:
        async with session.post(BASE_URL, headers=HEADERS, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["message"]["content"]

''' A function that runs a batch of prompts concurrently using asyncio and aiohttp.
It takes a model ID, system instruction, and a list of prompts, and returns the results.
It uses a semaphore to limit the number of concurrent requests.'''
async def run_batch(model: str, system_instruction: str, prompts: List[str]):
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_chat_completion(session, sem, model, system_instruction, prompt)
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
    prompt="Calculate PERC score for this patient."
):
    results = []
    wrong_outputs = []
    invalid_outputs = []
    parsed_results_per_model = {model_id: [] for model_id in model_ids}

    subset_df = perc_df.head(sample)
    iterate_df = perc_df if full_df else subset_df

    for model_id in model_ids:
        print(f"\n=== Evaluating model: {model_id} ===\n")
        wrong = invalid = correct = 0
        total = len(iterate_df)
        prompts = []

        for idx, row in iterate_df.iterrows():
            entities = entities = ast.literal_eval(row["Relevant Entities"]) if include_relevant_entities else None
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
            prompts=prompts
        )

        for i, reply in enumerate(results_async):
            progress = (i + 1) / len(results_async) * 100
            print(f"\rParsing: {progress:.1f}% ({i + 1}/{len(results_async)})", end="", flush=True)

            row = iterate_df.iloc[i]
            entities = row["Parsed Entities"]
            ground_truth = row["Ground Truth Answer"]

            parsed = extract_json_from_text(reply)
            if parsed is None or "Answer" not in parsed:
                # parsed = extract_score_from_text(reply) 
                if parsed is None or "Answer" not in parsed:
                    invalid += 1
                    invalid_outputs.append({
                        "model": model_id,
                        "note": row["Patient Note"],
                        "entities": entities if include_relevant_entities else None,
                        "reply": reply,
                        "reason": "Invalid JSON or missing 'Answer'"
                    })
                    parsed_results_per_model[model_id].append({
                        "note": row["Patient Note"],
                        "entities": entities if include_relevant_entities else None,
                        "ground_truth": ground_truth,
                        "parsed_criteria": None,
                        "predicted": None,
                        "raw_reply": reply,
                        "valid": False,
                        "correct": False
                    })
                    continue

            predicted_score = parsed.get("Answer")

            # Extract all criteria booleans except the 'Answer'
            criteria = {k: v for k, v in parsed.items() if k != "Answer"}

            if predicted_score == ground_truth:
                correct += 1
            else:
                wrong += 1
                wrong_outputs.append({
                    "model": model_id,
                    "note": row["Patient Note"],
                    "entities": entities if include_relevant_entities else None,
                    "reply": reply,
                    "expected": ground_truth,
                    "predicted": predicted_score
                })

            parsed_results_per_model[model_id].append({
                "note": row["Patient Note"],
                "entities": entities if include_relevant_entities else None,
                "ground_truth": ground_truth,
                "parsed_criteria": criteria,
                "predicted": predicted_score,
                "raw_reply": reply,
                "valid": True,
                "correct": predicted_score == ground_truth
            })

        valid = correct + wrong
        accuracy = correct / valid if valid > 0 else 0

        results.append({
            "model_id": model_id,
            "wrong": wrong,
            "correct": correct,
            "invalid": invalid,
            "total": total,
            "accuracy": round(accuracy, 3)
        })

        print("\nDone ✅")

    results_df = pd.DataFrame(results)
    print("\n=== Summary Table ===")
    print(results_df)

    return invalid_outputs, wrong_outputs, results_df, parsed_results_per_model


# %% [markdown]
# #### Troubleshooting wrong/invalid replies

# %%
def print_outputs(wrong_outputs,invalid_outputs):
    for i, entry in enumerate(wrong_outputs):
        print(f"\n--- Wrong Reply {i+1} ---")
        print(f"Expected: {entry['expected']} | Predicted: {entry['predicted']}")
        print(f"Reply:\n{entry['reply']}")

    # Extract invalid replies with metadata
    for i, entry in enumerate(invalid_outputs):
        print(f"\n--- Invalid Reply {i+1} ---")
        print(f"Expected: {entry['expected']} | Predicted: {entry['predicted']}")
        print(f"Reply:\n{entry['reply']}")



# %% [markdown]
# ### Accuracy (Used the data from "More detailed system instruction (does not include relevant entities in prompt)")

# %% [markdown]
# #### Outputs and Ground truth answer

# %%
''' A function that computes the truth table for the PERC criteria based on parsed JSON outputs and patient data.
It returns a DataFrame with accuracy for each criterion per model.'''
def compute_truth_table(parsed_json, patient_df):
    rows = []

    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ['true', 'yes', '1']
        return bool(val)

    for model_id, outputs in parsed_json.items():
        for i, output in enumerate(outputs):
            try:
                criteria_pred = output["parsed_criteria"]["Criteria"]
            except (TypeError, KeyError):
                continue  # skip bad or invalid outputs

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
                match = (pred_val == true_val)
                rows.append({
                    "Model": model_id,
                    "Criterion": criterion,
                    "Correct": match
                })

    df = pd.DataFrame(rows)

    # Group by model and criterion, compute mean accuracy
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
def plot_criteria_accuracy_by_outcome(parsed_json, patient_df):
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ['true', 'yes', '1']
        return bool(val)

    def compute_data(filter_correct: bool):
        rows = []
        for model_id, outputs in parsed_json.items():
            for i, output in enumerate(outputs):
                if output.get("correct") != filter_correct:
                    continue

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
                    match = (pred_val == true_val)
                    rows.append({
                        "Model": model_id,
                        "Criterion": criterion,
                        "CorrectCriterion": match
                    })

        return pd.DataFrame(rows)

    # Get data for both correct and incorrect predictions
    correct_df = compute_data(filter_correct=True)
    incorrect_df = compute_data(filter_correct=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for df, ax, title in zip(
        [correct_df, incorrect_df],
        axes,
        ["Given Correct Final Answer", "Given Incorrect Final Answer"]
    ):
        if df.empty:
            ax.set_visible(False)
            continue

        summary = (
            df.groupby(["Model", "Criterion"])["CorrectCriterion"]
            .mean()
            .reset_index()
            .pivot(index="Criterion", columns="Model", values="CorrectCriterion")
            .round(3)
        )

        sns.heatmap(summary, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=True, ax=ax)
        ax.set_title(f"Criterion Accuracy — {title}")
        ax.set_xlabel("Model")
        ax.set_ylabel("PERC Criterion")

    plt.tight_layout()
    plt.show()

'''Takes a parsed_json object (output from each model), Takes a patient_df (your DataFrame with patient notes and entities), Extracts and compares predicted vs actual PERC criteria, 
And produces pie charts showing how often models: Got all criteria right, Got one or more criteria wrong, Separately for correct and incorrect final predictions.'''
def plot_criteria_accuracy_pie(parsed_json, patient_df):
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

    # Store counts separately for correct/incorrect predictions
    model_stats = {}

    for model_id, outputs in parsed_json.items():
        correct_all = 0
        correct_some = 0
        wrong_all = 0
        wrong_some = 0

        for i, output in enumerate(outputs):
            try:
                criteria_pred = output["parsed_criteria"]["Criteria"]
            except (TypeError, KeyError):
                continue

            row = patient_df.iloc[i]
            entities = row["Parsed Entities"]
            truth = get_truth(entities)

            match_flags = []
            for criterion, true_val in truth.items():
                pred_val = criteria_pred.get(criterion)
                if pred_val is None:
                    continue
                match_flags.append(pred_val == true_val)

            if not match_flags:
                continue  # Skip if we couldn't match anything

            if builtins.all(match_flags):
                if output.get("correct"):
                    correct_all += 1
                else:
                    wrong_all += 1
            else:
                if output.get("correct"):
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

    # Plot 2 pie charts per model
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



def print_troubleshooting_outputs(wrong_outputs, invalid_outputs):
    """
    Prints wrong and invalid LLM outputs with expected and predicted values, and full replies.
    
    Args:
        wrong_outputs (list): List of dicts with keys 'expected', 'predicted', 'reply'.
        invalid_outputs (list): List of dicts with at least a 'reply' key.
    """
    # Print wrong outputs
    for i, entry in enumerate(wrong_outputs):
        print(f"\n--- Wrong Reply {i+1} ---")
        print(f"Expected: {entry.get('expected')} | Predicted: {entry.get('predicted')}")
        print(f"Reply:\n{entry.get('reply')}")

    # Print invalid outputs with metadata
    for i, entry in enumerate(invalid_outputs):
        print(f"\n--- Invalid Reply {i+1} ---")
        print(f"Expected: {entry.get('expected', 'N/A')} | Predicted: {entry.get('predicted', 'N/A')}")
        print(f"Reply:\n{entry.get('reply')}")

    # Extract and print replies only
    invalid_replies = [entry['reply'] for entry in invalid_outputs if 'reply' in entry]
    for i, reply in enumerate(invalid_replies):
        print(f"\n--- Invalid Reply (Only) {i+1} ---\n{reply}")



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


