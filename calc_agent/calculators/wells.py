from typing import Any, Dict
from .base import CalcResult
from ..calc_utils import to_float, as_bool

class WELLS:
    name = "wells"
    required_fields = [
        "Previously Documented Pulmonary Embolism",
        "Heart Rate or Pulse",
        "Immobilization for at least 3 days",
        "Hemoptysis",
        "Surgery in the previous 4 weeks",
        "Malignancy with treatment within 6 months or palliative",
        "Pulmonary Embolism is #1 diagnosis OR equally likely",
        "Previously documented Deep Vein Thrombosis"
    ]

    def normalize(self, e: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Previously Documented Pulmonary Embolism": as_bool(e.get("Previously Documented Pulmonary Embolism")),
            "Heart Rate or Pulse": to_float(e.get("Heart Rate or Pulse")),
            "Immobilization for at least 3 days": as_bool(e.get("Immobilization for at least 3 days")),
            "Hemoptysis": as_bool(e.get("Hemoptysis")),
            "Surgery in the previous 4 weeks": as_bool(e.get("Surgery in the previous 4 weeks")),
            "Malignancy with treatment within 6 months or palliative": as_bool(e.get("Malignancy with treatment within 6 months or palliative")),
            "Pulmonary Embolism is #1 diagnosis OR equally likely": as_bool(e.get("Pulmonary Embolism is #1 diagnosis OR equally likely")),
            "Previously documented Deep Vein Thrombosis": as_bool(e.get("Previously documented Deep Vein Thrombosis"))
        }

    def validate(self, x: Dict[str, Any]) -> list[str]:
        bad = []
        hr = x.get("Heart Rate or Pulse")
        if hr is None or not (20 <= hr <= 250):
            bad.append("Heart Rate or Pulse")
        return bad

    def compute(self, x: Dict[str, Any]) -> CalcResult:
        crit = {
            "Clinical signs of DVT": 3 if x["Previously documented Deep Vein Thrombosis"] else 0,
            "PE more likely than alternatives": 3 if x["Pulmonary Embolism is #1 diagnosis OR equally likely"] else 0,
            "HR > 100": 1.5 if x["Heart Rate or Pulse"] is not None and x["Heart Rate or Pulse"] > 100 else 0,
            "Recent immobilization or surgery": 1.5 if x["Immobilization for at least 3 days"] or x["Surgery in the previous 4 weeks"] else 0,
            "Previous DVT or PE": 1.5 if x["Previously Documented Pulmonary Embolism"] else 0,
            "Hemoptysis": 1 if x["Hemoptysis"] else 0,
            "Malignancy": 1 if x["Malignancy with treatment within 6 months or palliative"] else 0
        }
        total_score = sum(crit.values())

        if total_score > 6:
            interp = "High probability of PE"
        elif 2 <= total_score <= 6:
            interp = "Moderate probability of PE"
        else:
            interp = "Low probability of PE"

        return CalcResult(name=self.name, score=to_float(total_score), criteria=crit, interpretation=interp)
