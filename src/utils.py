import yaml
from typing import Dict, Any

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        return yaml.safe_load(file)