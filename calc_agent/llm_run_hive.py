from .calc_utils import *


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
    max_tokens=1000,
    perc_df=None
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
                entities=ast.literal_eval(row["Relevant Entities"]) if include_relevant_entities else None,
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
            entities = ast.literal_eval(row["Relevant Entities"]) if include_relevant_entities else ""
            ground_truth = row["Ground Truth Answer"]

            parsed = extract_json_from_text(reply)
            entry_type = "invalid"
            criteria = None
            predicted_score = None
            valid = False
            correct_flag = False

            rel_entities = ast.literal_eval(row["Relevant Entities"]) 

            if parsed is not None:
                criteria = {k: v for k, v in parsed.items() if k != "Answer"}
                if "Answer" in parsed:
                    try:
                        parsed["Answer"] = float(parsed["Answer"])
                    except Exception as e:
                        # print(f"Failed to convert Answer to int: {parsed['Answer']}, error: {e}")
                        parsed["Answer"] = None

                    answer = parsed.get("Answer")
                    if answer is not None:
                        predicted_score = answer
                        
                        valid = True
                        if row["Output Type"]=="integer":
                            if predicted_score == float(ground_truth):
                                entry_type = "correct"
                                correct_flag = True
                            else:
                                entry_type = "wrong"
                        else:
                            upper_limit = row["Upper Limit"]
                            lower_limit = row["Lower Limit"]
                            if float(lower_limit)<= predicted_score <= float(upper_limit):
                                entry_type = "correct"
                                correct_flag = True
                            else:
                                entry_type = "wrong"

            results_data.append({
                "model_id": model_id,
                "note": row["Patient Note"],
                "entities": rel_entities ,
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
