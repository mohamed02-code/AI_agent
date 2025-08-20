from typing import Any, Dict
from .base import CalcResult
from ..calc_utils import to_float, as_bool
from ..perc_utils import perc_score

class PERC:
    name = "perc"
    required_fields = [
        "age","heart_rate","oxygen_saturation",
        "has_hemoptysis","on_estrogen","history_dvt_pe",
        "unilateral_leg_swelling","recent_trauma_or_surgery"
    ]



    def normalize(self, e: Dict[str, Any]) -> Dict[str, Any]:
        return {k: to_float(e.get(k)) if k in ["age","heart_rate","oxygen_saturation"] else as_bool(e.get(k))
                for k in self.required_fields}

    def validate(self, x: Dict[str, Any]) -> list[str]:
        bad = []
        if x["age"] is None or not (0 <= x["age"] <= 120): bad.append("age")
        if x["heart_rate"] is None or not (20 <= x["heart_rate"] <= 250): bad.append("heart_rate")
        if x["oxygen_saturation"] is None or not (50 <= x["oxygen_saturation"] <= 100): bad.append("oxygen_saturation")
        return bad

    def compute(self, x: Dict[str, Any]) -> CalcResult:
        s = perc_score(*[x[f] for f in self.required_fields])
        crit = {
            "Age < 50": x["age"] is not None and x["age"] < 50,
            "HR < 100": x["heart_rate"] is not None and x["heart_rate"] < 100,
            "O₂ ≥ 95%": x["oxygen_saturation"] is not None and x["oxygen_saturation"] >= 95,
            "No hemoptysis": not x["has_hemoptysis"],
            "No Hormone use": not x["on_estrogen"],
            "No prior VTE or DVT": not x["history_dvt_pe"],
            "No unilateral leg swelling": not x["unilateral_leg_swelling"],
            "No recent trauma or surgery": not x["recent_trauma_or_surgery"],
        }
        interp = "PERC negative (if low pretest probability)" if s == 0 else "Not PERC negative" if s is not None else None
        return CalcResult(name=self.name, score=s, criteria=crit, interpretation=interp)


