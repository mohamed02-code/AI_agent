# Medical Calculators with LLMs
Paper: https://www.nature.com/articles/s41746-025-01475-8 

Dataset: https://huggingface.co/datasets/ncbi/MedCalc-Bench-v1.0

    pip install -r requirements.txt

# CalculatorAgent  Usage

This module provides functionality to calculate clinical scores (like **PERC** and **Wells**) and optionally run LLM-based evaluation for patient notes. It supports both **direct calculation** from extracted entities and **LLM-enhanced prompts**.

## Using `CalculatorAgent` with LLMs

`CalculatorAgent` can run models to extract entities and compute scores automatically.

### Initialization

```python
dataset = load_dataset("ncbi/MedCalc-Bench-v1.0")
df = dataset["train"].to_pandas()

agent = CalculatorAgent(
    model_ids=["Llama-3.3-70B-Instruct"],
    calculator_name="perc",
    df=df,
    prompt_mode="calculator",  # use "enhanced_prompt" for full LLM scoring
    prompt_config_path="prompts.json"
)
```

### Running the Agent

```python
import asyncio

results_df = asyncio.run(agent.run())
print(results_df.head())
```

**Columns in `results_df`:**

* `model_id`: LLM model used
* `reply`: raw LLM output
* `inputs`: normalized extracted entities
* `calc_param`: parsed entities from LLM
* `calculated_score`: score computed by calculator
* `status_recomputed`: `"Correct"`, `"Incorrect"`, or `"Invalid"`
* `ground_truth`: reference score from dataset

---

## Supported Calculators

* **PERC or PERC Rule for Pulmonary Embolism ** (`perc`, `perc rule`)  
* **Wells or Wells' Criteria for Pulmonary Embolism** (`wells`, `wells score`)


---

## Handling Multiple Models

`CalculatorAgent` supports multiple `model_ids`. Results are returned for each model separately, and you can filter by `model_id`:

```python
results_df[results_df["model_id"] == "Llama-3.3-70B-Instruct"]
```

---

## Notes

* `prompt_mode="calculator"`: Only extract entities with selected model and compute score automatically.
* `prompt_mode="enhanced_prompt"`: Run full LLM evaluation with scoring logic embedded in prompt.


