import json
import pandas as pd
from typing import List, Any,Union
from .calc_utils import *
from .llm_run_hive import *
from .perc_utils import *

# agent.py
from typing import Dict, Any
from .calculators.base import Calculator, CalcResult
from .calculators.perc import PERC
from .calculators.wells import WELLS
from .calculators.CreatinineClearance import CreatinineClearance

REGISTRY: Dict[str, object] = {
    "perc": PERC(),
    "wells": WELLS(),
    "creatinine_clearance": CreatinineClearance()
}

ALIASES = {
    "perc": {"perc", "perc rule"},
    "wells": {"wells", "wells score"},
    "creatinine_clearance": {
        "cc", 
        "cockcroft gault", 
        "creatinine clearance", 
        "c-g equation", 
        "cockcroft–gault equation"
    }
}


def resolve_calculator(name: str) -> Union[str, None]:
    """
    resolve takes in a string and returns chosen calculator name based on ALIASES
    """
    n = (name or "").strip().lower()
    for key, names in ALIASES.items():
        if n in names:
            return key
    return REGISTRY.get(n) and n




def run_calculator(name: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes in a calculator names and checks if it exists and calls it by inputting entities
    """
    key = resolve_calculator(name)
    if not key:
        return {"ok": False, "error": f"Unknown calculator '{name}'", "available": list(REGISTRY.keys())}

    calc = REGISTRY[key]
    inputs = calc.normalize(entities)
    missing = calc.validate(inputs)
    if missing:
        return {"ok": False, "calculator": key, "error": f"Missing/invalid: {', '.join(missing)}", "inputs": inputs}

    result: CalcResult = calc.compute(inputs)
    return {
        "ok": result.score is not None,
        "calculator": result.name,
        "score": result.score,
        "criteria": result.criteria,
        "interpretation": result.interpretation,
        "inputs": inputs
    }

class LLMResponse:
    def __init__(self, text: str, raw: Any = None):
        self.text = text
        self.raw = raw


class CalculatorAgent:
    """
    Caluclator Agent runs a model using saved prompts and parses outputs for scores/entities depending on calculator
    """
    def __init__(
        self,
        model_ids: List[str],
        calculator_name: str,
        df: pd.DataFrame,
        prompt_mode: str = "enhanced_prompt",
        prompt_config_path: str = "prompts.json", ## TODO change how path is set/handled
        prompt: str = ""
    ):
        self.model_ids = model_ids
        self.calculator_name = calculator_name.lower()
        self.df = df
        self.prompt_mode = prompt_mode
        self.calc: Calculator | None = REGISTRY.get(self.calculator_name)
        # load prompts for LLM modes

        with open(prompt_config_path, "r") as f:
            self.prompt_config = json.load(f)
        if self.calculator_name not in self.prompt_config:
            raise ValueError(f"No prompts found for calculator '{self.calculator_name}'")
        self.sys_instruct = self.prompt_config[self.calculator_name]["sys_instruct"]
        if self.prompt_mode != "calculator":
            self.sys_instruct = self.prompt_config[self.calculator_name]["sys_instruct"]
            # use prompt provided instead of enhanced prompt
            if prompt:
                self.prompt = prompt
            else:
                self.prompt = self.prompt_config[self.calculator_name][self.prompt_mode]
        else:
            # use `extract_entities` field from prompt_config for calculator mode
            if "extract_entities"not in self.prompt_config[self.calculator_name]:
                raise ValueError(f"Calculator not supported yet. '{self.calculator_name}'")
            self.prompt = self.prompt_config[self.calculator_name]["extract_entities"]

    async def run(self, use_df: pd.DataFrame = None,sample=0) -> pd.DataFrame:
        if self.prompt_mode == "calculator": 
            if not self.calc:
                raise ValueError(f"Calculator '{self.calculator_name}' not found in registry.")

            df_to_use = use_df if use_df is not None else self.df
            results = []
            # Step 1: run model to extract entities as JSON
            model_resp = await run_models_with_output(
                        sys_instruct=self.sys_instruct,
                        prompt=self.prompt,
                        model_ids=self.model_ids,
                        include_relevant_entities=False,
                        full_df=True if sample==0 else False,
                        max_tokens=1000,
                        temperature=0.0,
                        perc_df=use_df,sample=sample
                    )
            for _, row in model_resp.iterrows():
                patient_text = row.get("note", "")  

                try:
                    entities = row.get("parsed_criteria")  # use the current row’s value
                    if isinstance(entities, str):  # if it’s a string, parse JSON
                        entities = ast.literal_eval(entities)
                    if not isinstance(entities, dict):  # safety net
                        entities = {}
                except Exception:
                    entities = {}


                # Step 2: normalize, validate, compute
                inputs = self.calc.normalize(entities)
                missing = self.calc.validate(inputs)
                if missing:
                    
                    calc_result = None
                    status = "Invalid"
                else:
                    calc_result = self.calc.compute(inputs)
                    status = "OK"

                results.append({
                    "model_id": row.get("model_id"),
                    "reply": row.get("reply", ""),
                    "inputs": inputs,
                    "calc_param": entities,
                    "calculated_score": calc_result.score if calc_result else None,
                    "status_recomputed": status,
                    "ground_truth": row.get("ground_truth"),
                    "abs_difference": 0 ## TODO add more fields to analyze floating point data
                })

            results_df = pd.DataFrame(results)

            # Step 3: classify correctness
            def classify_row(row):
                cs = row.get("calculated_score")
                gt = row.get("ground_truth")
                # mark as invalid if missing
                if pd.isna(cs) or gt is None:
                    return "Invalid"

                try:
                    if row["Output Type"]=="integer":
                        cs_val = float(cs)
                        gt_val = float(gt)
                        return "Correct" if int(round(cs_val)) == int(round(gt_val)) else "Incorrect"
                    else:
                        cs_val = float(cs)
                        gt_val = float(gt)
                        ulimit = gt_val*1.05
                        llimit = gt_val*0.95
                        return "Correct" if llimit <= cs_val <= ulimit else "Incorrect"

                except Exception:
                    return "Invalid"
                


            results_df["status_recomputed"] = results_df.apply(classify_row, axis=1) 
        # fallback: run full LLM prompt
        else:
            results_df = await run_models_with_output(
                sys_instruct=self.sys_instruct,
                prompt=self.prompt,
                model_ids=self.model_ids,
                include_relevant_entities=False,
                full_df= True if sample==0 else False,
                max_tokens=1000,
                temperature=0.0,
                perc_df=use_df,sample=sample
            )
        return results_df

