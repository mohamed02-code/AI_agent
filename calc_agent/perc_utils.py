from .calc_utils import *
from .llm_run_hive import *

# Load dataset
# dataset = load_dataset("ncbi/MedCalc-Bench-v1.0")
# df = dataset["train"].to_pandas()  # or "test"
# df_test = dataset["test"].to_pandas()  # or "test"

# perc_df = df[df["Calculator Name"] == "PERC Rule for Pulmonary Embolism"]

# perc_df_test = df_test[df_test["Calculator Name"] == "PERC Rule for Pulmonary Embolism"]

# # Load dataset
# dataset = load_dataset("ncbi/MedCalc-Bench-v1.0")
# df = dataset["train"].to_pandas()  # or "test"
# df_test = dataset["test"].to_pandas()  # or "test"

# perc_df = df[df["Calculator Name"] == "PERC Rule for Pulmonary Embolism"]

# perc_df_test = df_test[df_test["Calculator Name"] == "PERC Rule for Pulmonary Embolism"]


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


# Models that I will be using
model_ids = [
    "Llama-3.3-70B-Instruct",
    "Med42-Qwen2.5-72B-v3-bi"
]


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



def add_calc_param_to_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structured calculator parameters from each model reply into a new column.

    Parameters
    ----------
    results_df : pd.DataFrame
        Input dataframe that MUST contain a column:
          - 'reply' : str or None
            Raw LLM response text potentially containing a JSON object with calculator fields.

    Returns
    -------
    pd.DataFrame
        A copy of `results_df` with an added column:
          - 'calc_param' : dict | None
            The parsed JSON object extracted from 'reply' if present and valid; otherwise None.

    Notes
    -----
    - This function does not validate the semantic correctness of fields; it only parses JSON out of text.
    - Downstream code can feed 'calc_param' into `add_calculated_score` or `verify_calc_json`.
    """
    df = results_df.copy()

    def safe_extract_json(reply: Any) -> Optional[Dict[str, Any]]:
        try:
            return extract_json_from_text(reply) if isinstance(reply, str) else None
        except Exception:
            return None

    df["calc_param"] = df["reply"].apply(safe_extract_json)
    return df


def add_calculated_score(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the PERC score from parsed calculator parameters and store it in a new column.

    Parameters
    ----------
    results_df : pd.DataFrame
        Input dataframe that MUST contain a column:
          - 'calc_param' : dict | None
            A dictionary with PERC fields, typically produced by `add_calc_param_to_results`.

            Expected keys inside 'calc_param' (all optional; missing treated as None/False):
              - 'age' : float|int|str
              - 'heart_rate' : float|int|str
              - 'oxygen_saturation' : float|int|str
              - 'has_hemoptysis' : bool|int|str
              - 'on_estrogen' : bool|int|str
              - 'history_dvt_pe' : bool|int|str
              - 'unilateral_leg_swelling' : bool|int|str
              - 'recent_trauma_or_surgery' : bool|int|str

    Returns
    -------
    pd.DataFrame
        A copy of `results_df` with an added column:
          - 'calculated_score' : pd.Series(dtype='Int64')
            The integer PERC score recomputed from 'calc_param' (0–8). Missing/invalid rows are <NA>.

    Notes
    -----
    - Numeric-like strings are coerced with `to_float`.
    - Booleans accept common truthy/falsey values when originally produced by the model.
    """
    df = results_df.copy()

    def compute_score(params: Any) -> Optional[int]:
        if not isinstance(params, dict):
            return None
        return perc_score(
            to_float(params.get("age")),
            to_float(params.get("heart_rate")),
            to_float(params.get("oxygen_saturation")),
            bool(params.get("has_hemoptysis", False)),
            bool(params.get("on_estrogen", False)),
            bool(params.get("history_dvt_pe", False)),
            bool(params.get("unilateral_leg_swelling", False)),
            bool(params.get("recent_trauma_or_surgery", False)),
        )

    df["calculated_score"] = df["calc_param"].apply(compute_score).astype("Int64")
    return df


def norm_key(k: Any) -> Any:
    """
    Normalize an entity key to a lowercase ASCII-like form suitable for matching.

    Parameters
    ----------
    k : Any
        Original dictionary key. If not a str, the value is returned unchanged.

    Returns
    -------
    Any
        Normalized string key (lowercased, common unicode digits/O₂ simplified),
        or original value if non-string.
    """
    if not isinstance(k, str):
        return k
    k = k.strip().lower()
    # Basic normalizations for common unicode glyphs found in clinical text
    k = k.replace('o₂', 'o2').replace('₀', '0').replace('₁', '1').replace('₂', '2')
    return k


# Typed alias map for canonical keys -> acceptable aliases in incoming data
KEY_ALIASES: Dict[str, List[str]] = {
    # vitals
    'age': ['age'],
    'heart_rate': ['heart rate or pulse', 'heart rate', 'pulse'],
    'o2_saturation': ['o2 saturation percentage', 'oxygen saturation', 'spo2', 'o2 sat'],
    # binary criteria
    'hemoptysis': ['hemoptysis'],
    'hormone_use': ['hormone use', 'estrogen use', 'on estrogen'],
    'prior_vte': [
        'prior vte', 'previously documented pulmonary embolism',
        'previously documented deep vein thrombosis', 'history dvt pe',
        'previously documented dvt', 'previously documented pe'
    ],
    'unilateral_leg_swelling': ['unilateral leg swelling'],
    'recent_trauma_or_surgery': ['recent surgery or trauma', 'recent trauma or surgery'],
}


def pick(entities: Mapping[str, Any], canonical: str) -> Any:
    """
    Retrieve the first available value for a canonical field using alias matching.

    Parameters
    ----------
    entities : Mapping[str, Any]
        Dictionary of (normalized) entity keys to values.
    canonical : str
        Canonical field name (e.g., 'heart_rate', 'prior_vte').

    Returns
    -------
    Any
        The first matched value among the canonical key and its aliases (case-insensitive),
        or None if no match is found.

    Notes
    -----
    - This function attempts three lookups per alias: as-is, Title Case, and UPPER.
    - For best results, pass in a dict whose keys were preprocessed by `norm_key`.
    """
    targets = [canonical] + KEY_ALIASES.get(canonical, [])
    for t in targets:
        v = entities.get(t) or entities.get(t.title()) or entities.get(t.upper())
        if v is not None:
            return v
    return None


def normalize_entities(entities: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize heterogeneous entity dictionaries into a single canonical PERC input schema.

    Parameters
    ----------
    entities : dict | None
        Raw entity dictionary extracted from notes or model output. Keys can vary in spelling,
        case, and unicode. Values may be numbers, strings, booleans, or lists.

    Returns
    -------
    dict
        Canonical dictionary with keys:
          - 'age' : float | None
          - 'heart_rate' : float | None
          - 'o2_saturation' : float | None
          - 'has_hemoptysis' : bool
          - 'on_estrogen' : bool
          - 'history_dvt_pe' : bool
          - 'unilateral_leg_swelling' : bool
          - 'recent_trauma_or_surgery' : bool

    Notes
    -----
    - Numeric-like strings/lists are coerced using `to_float`.
    - Booleans accept common textual variants ('yes', 'true', 'present', '1', etc.).
    """
    out: Dict[str, Any] = {}
    tmp = {norm_key(k): v for k, v in (entities or {}).items()}

    def as_bool(x: Any) -> bool:
        if isinstance(x, list) and x:
            x = x[0]
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            return x.strip().lower() in {'true', 'yes', '1', 'present', 'y'}
        return False

    out['age'] = to_float(pick(tmp, 'age'))
    out['heart_rate'] = to_float(pick(tmp, 'heart_rate'))
    out['o2_saturation'] = to_float(pick(tmp, 'o2_saturation'))

    out['has_hemoptysis'] = as_bool(pick(tmp, 'hemoptysis'))
    out['on_estrogen'] = as_bool(pick(tmp, 'hormone_use'))
    out['history_dvt_pe'] = as_bool(pick(tmp, 'prior_vte'))
    out['unilateral_leg_swelling'] = as_bool(pick(tmp, 'unilateral_leg_swelling'))
    out['recent_trauma_or_surgery'] = as_bool(pick(tmp, 'recent_trauma_or_surgery'))
    return out


def verify_calc_json(obj: Any) -> Dict[str, Any]:
    """
    Verify a calculator JSON object by recomputing the PERC score from its fields.

    Parameters
    ----------
    obj : dict | Any
        JSON-like dictionary expected to contain:
          - 'age', 'heart_rate', 'oxygen_saturation' : numeric-like
          - 'has_hemoptysis', 'on_estrogen', 'history_dvt_pe',
            'unilateral_leg_swelling', 'recent_trauma_or_surgery' : bool-like
          - 'Answer' : int-like (claimed PERC score)

    Returns
    -------
    dict
        Verification report with keys:
          - 'ok' : bool
              True if recomputed score exactly matches the claimed 'Answer'.
          - 'claimed' : int | None
              The claimed score extracted from `obj['Answer']` if present/parseable.
          - 'recomputed' : int | None
              Score recomputed by `perc_score` from the provided fields.
          - 'reason' : str (optional)
              Present only if an exception occurred (e.g., missing/invalid fields).

    Notes
    -----
    - This check is purely programmatic; it does not use an LLM.
    - If mandatory numeric fields are missing (age, heart_rate, oxygen_saturation),
      `perc_score` may return None and the verification will fail.
    """
    if not isinstance(obj, dict):
        return {"ok": False, "reason": "not a dict"}
    try:
        recomputed = perc_score(
            to_float(obj.get("age")),
            to_float(obj.get("heart_rate")),
            to_float(obj.get("oxygen_saturation")),
            bool(obj.get("has_hemoptysis", False)),
            bool(obj.get("on_estrogen", False)),
            bool(obj.get("history_dvt_pe", False)),
            bool(obj.get("unilateral_leg_swelling", False)),
            bool(obj.get("recent_trauma_or_surgery", False)),
        )
        claimed_raw = obj.get("Answer")
        claimed: Optional[int] = None
        if claimed_raw is not None:
            try:
                claimed = int(claimed_raw)
            except Exception:
                claimed = None
        return {
            "ok": (claimed is not None and recomputed == claimed),
            "claimed": claimed,
            "recomputed": recomputed,
        }
    except Exception as e:
        return {"ok": False, "reason": repr(e)}


def compute_perc_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy using programmatically recomputed PERC scores and plot pies.

    Pipeline
    --------
    results_df  --(add_calc_param_to_results)-->  calc_param (dict)
                 --(add_calculated_score)------->  calculated_score (Int64)
                 --(verify_calc_json)----------->  verify (dict with ok/claimed/recomputed)
                 --(classify)------------------->  status_recomputed ("correct"/"incorrect"/"invalid")

    Classification logic
    --------------------
    - invalid  : calculated_score is <NA>/None or ground_truth is missing/non-castable
    - correct  : int(calculated_score) == int(ground_truth)
    - incorrect: otherwise

    Plots
    -----
    - One overall pie (all models combined)
    - One pie per model_id

    Parameters
    ----------
    results_df : pd.DataFrame
        Output dataframe from `run_models_with_output(...)`.
        Must contain at least: ['reply', 'ground_truth', 'model_id'].

    Returns
    -------
    pd.DataFrame
        A summary table with counts per model for
        ['Correct', 'Incorrect', 'Invalid', 'Total'] and overall row ('__ALL__').
    """
    # Step 1: parse JSON from replies
    df = add_calc_param_to_results(results_df)

    # Step 2: recompute score in Python from parsed fields
    df = add_calculated_score(df)

    # Step 3: optional: store verification object (claimed vs recomputed)
    df["verify"] = df["calc_param"].apply(verify_calc_json)

    # Step 4: classify using recomputed score vs ground truth
    def classify_row(row):
        cs = row.get("calculated_score")
        gt = row.get("ground_truth")
        if pd.isna(cs) or gt is None:
            return "Invalid"
        try:
            return "Correct" if int(cs) == int(gt) else "Incorrect"
        except Exception:
            return "Invalid"

    df["status_recomputed"] = df.apply(classify_row, axis=1)

    # Build summary (per-model + overall)
    def counts_for(group: pd.DataFrame) -> Dict[str, int]:
        c = group["status_recomputed"].value_counts()
        correct = int(c.get("Correct", 0))
        incorrect = int(c.get("Incorrect", 0))
        invalid = int(c.get("Invalid", 0))
        total = correct + incorrect + invalid
        return {"Correct": correct, "Incorrect": incorrect, "Invalid": invalid, "Total": total}

    per_model = (
        df.groupby("model_id", dropna=False)
          .apply(counts_for)
          .apply(pd.Series)
          .reset_index()
          .rename(columns={"index": "model_id"})
    )

    overall = counts_for(df)
    overall_row = pd.DataFrame([{"model_id": "__ALL__", **overall}])

    summary = pd.concat([overall_row, per_model], ignore_index=True)


    return summary

def plot_pie_computed_results(summary_df: pd.DataFrame):
    # Build summary (per-model + overall)
    def counts_for(group: pd.DataFrame) -> Dict[str, int]:
        c = group["status_recomputed"].value_counts()
        correct = int(c.get("Correct", 0))
        incorrect = int(c.get("Incorrect", 0))
        invalid = int(c.get("Invalid", 0))
        total = correct + incorrect + invalid
        return {"Correct": correct, "Incorrect": incorrect, "Invalid": invalid, "Total": total}
    overall = counts_for(summary_df)
    per_model = (
        summary_df.groupby("model_id", dropna=False)
          .apply(counts_for)
          .apply(pd.Series)
          .reset_index()
          .rename(columns={"index": "model_id"})
    )
    # ----- Plotting -----
    # Colors (keep consistent)
    colors = ["mediumseagreen", "tomato", "slategray"]

    # 1) Overall pie
    fig_rows = max(1, len(per_model))  # at least 1 row of subplots section
    plt.figure(figsize=(6, 6))
    overall_vals = [overall["Correct"], overall["Incorrect"], overall["Invalid"]]
    if sum(overall_vals) == 0:
        plt.title("Overall: No data to plot")
    else:
        plt.pie(overall_vals, labels=["Correct", "Incorrect", "Invalid"],
                autopct="%1.1f%%", startangle=140, colors=colors)
        plt.title("Overall Accuracy (Recomputed vs Ground Truth)")
    plt.show()

    # 2) Per-model pies
    if len(per_model) > 0:
        fig, axes = plt.subplots(len(per_model), 1, figsize=(7, 5 * len(per_model)))
        if len(per_model) == 1:
            axes = [axes]  # normalize to list

        for ax, (_, row) in zip(axes, per_model.iterrows()):
            vals = [row["Correct"], row["Incorrect"], row["Invalid"]]
            if sum(vals) == 0:
                ax.axis("off")
                ax.set_title(f"{row['model_id']}: No data")
                continue
            ax.pie(vals, labels=["Correct", "Incorrect", "Invalid"],
                   autopct="%1.1f%%", startangle=140, colors=colors)
            ax.set_title(f"Model: {row['model_id']} (Recomputed vs Ground Truth)")
        plt.tight_layout()
        plt.show()


def plot_criteria_accuracy_heatmap(results_df: pd.DataFrame):
    """
    Plot criterion-level accuracy per model (ignores final-answer correctness).

    Rationale
    ---------
    In the current setup you ask models ONLY for the per-criterion booleans and
    not the final PERC Answer. That means the 'correct'/'invalid' labels tied to
    final answers are not meaningful. Instead, we compute accuracy for each of
    the 8 PERC criteria by comparing the model's predicted booleans against the
    ground-truth flags derived from the row's 'entities'.

    Requirements
    ------------
    - results_df must contain:
        - 'model_id'
        - 'parsed_criteria'  (dict with key 'Criteria' -> dict of criterion->bool)
        - 'entities'         (raw entity dict; we'll normalize it)

    Uses helpers (already defined in your codebase):
        - normalize_entities(entities: dict) -> dict with canonical fields
          ['age','heart_rate','o2_saturation','has_hemoptysis','on_estrogen',
           'history_dvt_pe','unilateral_leg_swelling','recent_trauma_or_surgery']

    Output
    ------
    - Displays a seaborn heatmap of accuracy by (Criterion x Model).
    - Returns the underlying accuracy DataFrame for further use.
    """


    # Consistent criterion labels (match what models output under "Criteria")
    CRITERIA_ORDER = [
        "Age < 50",
        "HR < 100",
        "O₂ ≥ 95%",
        "No hemoptysis",
        "No Hormone use",
        "No prior VTE or DVT",
        "No unilateral leg swelling",
        "No recent trauma or surgery",
    ]

    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in {"true", "yes", "1", "present", "y"}
        return False

    def truth_flags_from_entities(entities: dict) -> dict:
        """
        Build ground-truth PERC pass/fail flags from raw entities using your
        normalize_entities helper. Flags here are the *pass* versions to
        match the expected 'Criteria' keys (e.g., "Age < 50": True means 'passes').
        """
        e = normalize_entities(entities or {})

        age = e.get("age")
        hr = e.get("heart_rate")
        o2 = e.get("o2_saturation")

        return {
            "Age < 50": (age is not None and age < 50),
            "HR < 100": (hr is not None and hr < 100),
            "O₂ ≥ 95%": (o2 is not None and o2 >= 95),
            "No hemoptysis": (not e.get("has_hemoptysis", False)),
            "No Hormone use": (not e.get("on_estrogen", False)),
            "No prior VTE or DVT": (not e.get("history_dvt_pe", False)),
            "No unilateral leg swelling": (not e.get("unilateral_leg_swelling", False)),
            "No recent trauma or surgery": (not e.get("recent_trauma_or_surgery", False)),
        }

    # Collect row-wise comparisons (pred vs truth) for each criterion
    rows = []
    for _, row in results_df.iterrows():
        model_id = row.get("model_id")
        pc = row.get("parsed_criteria")

        # Get predicted criteria dict
        try:
            criteria_pred = pc.get("Criteria") if isinstance(pc, dict) else None
        except Exception:
            criteria_pred = None

        if not isinstance(criteria_pred, dict):
            continue  # skip if we don't have a predicted criteria dict

        # Build truth flags from entities
        truth = truth_flags_from_entities(row.get("entities", {}))

        # Compare criterion-by-criterion
        for crit in CRITERIA_ORDER:
            true_val = truth.get(crit, None)
            pred_val = criteria_pred.get(crit, None)
            if true_val is None or pred_val is None:
                # If either side is missing, skip this criterion for this row
                continue
            match = (to_bool(pred_val) == bool(true_val))
            rows.append({"Model": model_id, "Criterion": crit, "Correct": match})

    if not rows:
        print("No comparable criterion-level data available to plot.")
        return pd.DataFrame()

    df_comp = pd.DataFrame(rows)

    # Aggregate to accuracy (mean of boolean correctness)
    acc = (
        df_comp.groupby(["Model", "Criterion"])["Correct"]
        .mean()
        .reset_index()
    )

    # Pivot for heatmap
    acc_pivot = acc.pivot(index="Criterion", columns="Model", values="Correct")

    # Ensure consistent row order (optional)
    existing_rows = [c for c in CRITERIA_ORDER if c in acc_pivot.index]
    acc_pivot = acc_pivot.loc[existing_rows]

    # Plot
    # plt.figure(figsize=(1.8 * max(6, len(acc_pivot.columns)), 2.0 * len(acc_pivot.index)))
    plt.figure(figsize=(12, 4))
    sns.heatmap(
        acc_pivot.round(3),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=True
    )
    plt.title("PERC Criterion Accuracy by Model (criterion-only setup)")
    plt.xlabel("Model")
    plt.ylabel("PERC Criterion")
    plt.tight_layout()
    plt.show()

    return acc_pivot
def plot_criteria_accuracy_by_outcome(results_df: pd.DataFrame):
    """
    Plot criterion-level accuracy per model, separately for all, correct, and wrong final answers.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    CRITERIA_ORDER = [
        "Age < 50",
        "HR < 100",
        "O₂ ≥ 95%",
        "No hemoptysis",
        "No Hormone use",
        "No prior VTE or DVT",
        "No unilateral leg swelling",
        "No recent trauma or surgery",
    ]

    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in {"true", "yes", "1", "present", "y"}
        return False

    def truth_flags_from_entities(entities: dict) -> dict:
        e = normalize_entities(entities or {})
        age = e.get("age")
        hr = e.get("heart_rate")
        o2 = e.get("o2_saturation")
        return {
            "Age < 50": (age is not None and age < 50),
            "HR < 100": (hr is not None and hr < 100),
            "O₂ ≥ 95%": (o2 is not None and o2 >= 95),
            "No hemoptysis": (not e.get("has_hemoptysis", False)),
            "No Hormone use": (not e.get("on_estrogen", False)),
            "No prior VTE or DVT": (not e.get("history_dvt_pe", False)),
            "No unilateral leg swelling": (not e.get("unilateral_leg_swelling", False)),
            "No recent trauma or surgery": (not e.get("recent_trauma_or_surgery", False)),
        }

    # Helper to create comparison DataFrame
    def build_comparison_df(df_subset):
        rows = []
        for _, row in df_subset.iterrows():
            model_id = row.get("model_id")
            pc = row.get("parsed_criteria")
            try:
                criteria_pred = pc.get("Criteria") if isinstance(pc, dict) else None
            except Exception:
                criteria_pred = None
            if not isinstance(criteria_pred, dict):
                continue
            truth = truth_flags_from_entities(row.get("entities", {}))
            for crit in CRITERIA_ORDER:
                true_val = truth.get(crit)
                pred_val = criteria_pred.get(crit)
                if true_val is None or pred_val is None:
                    continue
                rows.append({"Model": model_id, "Criterion": crit, "Correct": (to_bool(pred_val) == bool(true_val))})
        return pd.DataFrame(rows)

    # Split by all / correct / wrong
    subsets = {
        "All": results_df,
        "Correct Final Answer": results_df[results_df.get("correct") == True],
        "Wrong Final Answer": results_df[results_df.get("correct") == False],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, (title, subset_df) in zip(axes, subsets.items()):
        comp_df = build_comparison_df(subset_df)
        if comp_df.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(title)
            continue
        acc = comp_df.groupby(["Criterion", "Model"])["Correct"].mean().reset_index()
        acc_pivot = acc.pivot(index="Criterion", columns="Model", values="Correct")
        existing_rows = [c for c in CRITERIA_ORDER if c in acc_pivot.index]
        acc_pivot = acc_pivot.loc[existing_rows]
        sns.heatmap(
            acc_pivot.round(3),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=(ax==axes[2]),  # show colorbar only on the last heatmap
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel("PERC Criterion")

    plt.show()

def plot_predicted_distribution(df, model_id):
    """
    Plots a distribution graph (histogram + KDE) of predicted scores for a given model_id.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing results_data.
        model_id (str): The model_id to filter on.
    """
    # filter df by model_id
    model_df = df[df["model_id"] == model_id]
    
    if model_df.empty:
        print(f"No data found for model_id = {model_id}")
        return
    
    plt.figure(figsize=(8, 5))
    
    sns.histplot(model_df["predicted"], kde=True, bins=20, color="skyblue", edgecolor="black")
    
    plt.title(f"Distribution of Predicted Scores for {model_id}", fontsize=14)
    plt.xlabel("Predicted Score")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()

