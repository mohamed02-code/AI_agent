from dataclasses import dataclass
from typing import Dict, Any, Union, Protocol

@dataclass
class CalcResult:
    name: str
    score: Union[int, float, None]
    criteria: Union[Dict[str, bool], None]
    interpretation: Union[str, None]

class Calculator(Protocol):
    name: str
    required_fields: list[str]
    def normalize(self, entities: Dict[str, Any]) -> Dict[str, Any]: ...
    def validate(self, inputs: Dict[str, Any]) -> list[str]: ...
    def compute(self, inputs: Dict[str, Any]) -> CalcResult: ...


