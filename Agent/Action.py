from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Action(BaseModel):
    model_config = {
        "json_schema_extra": {
            "properties": {
                "name": {"description": "Tool name"},
                "args": {"description": "Tool input arguments, containing arguments names and values"}
            }
        }
    }
    
    name: str
    args: Optional[Dict[str, Any]] = None

    def __str__(self):
        ret = f"Action(name={self.name}"
        if self.args:
            for k, v in self.args.items():
                ret += f", {k}={v}"
        ret += ")"
        return ret
