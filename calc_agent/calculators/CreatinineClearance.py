from typing import Any, Dict
from .base import CalcResult
from ..calc_utils import to_float

class CreatinineClearance:
    name = "creatinine_clearance_cockcroft_gault"
    required_fields = ["sex", "age", "weight", "creatinine"]

    def normalize(self, e: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sex": str(e.get("sex", "")).lower(),
            "age": to_float(e.get("age")[0] if isinstance(e.get("age"), (list, tuple)) else e.get("age")),
            "weight": to_float(e.get("weight")[0] if isinstance(e.get("weight"), (list, tuple)) else e.get("weight")),
            "creatinine": to_float(e.get("creatinine")[0] if isinstance(e.get("creatinine"), (list, tuple)) else e.get("creatinine")),
        }

    def validate(self, x: Dict[str, Any]) -> list[str]:
        bad = []
        if x.get("age") is None :
            bad.append("age")
        if x.get("weight") is None :
            bad.append("weight")
        if x.get("creatinine") is None :
            bad.append("creatinine")
        if x.get("sex") not in ["male", "female"]:
            bad.append("sex")
        return bad

    def compute(self, x: Dict[str, Any]) -> CalcResult:
        age = x["age"]
        weight = x["weight"]
        scr = x["creatinine"]  # mg/dL
        sex = x["sex"]

        # Cockcroft–Gault formula (no adjustment for body surface area)
        crcl = ((140 - age) * weight) / (72 * scr)
        if sex == "female":
            crcl *= 0.85

        crit = {
            "Age": age,
            "Weight (kg)": weight,
            "Serum Creatinine (mg/dL)": scr,
            "Sex": sex,
            "Formula Used": "Cockcroft–Gault Equation"
        }

        interp = f"Estimated Creatinine Clearance: {crcl:.2f} mL/min"

        return CalcResult(
            name=self.name,
            score=to_float(crcl),
            criteria=crit,
            interpretation=interp
        )
