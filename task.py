# dsl.py
from dataclasses import dataclass
from typing import List, Dict
import json
from pathlib import Path

def program_length(program: str) -> int:
    return len([tok for tok in program.strip().split() if tok])


def load_towers_config(path: Path) -> Dict[str, Dict]:
    """
    The data format is like:
    {
      "CL": {
        "correct_programs": [All programs could build this task]
      },
      ...
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    towers_cfg: Dict[str, Dict] = {}
    for item in raw:
        tower_id = item["towers"]
        program = item["program"]

        if tower_id not in towers_cfg:
            towers_cfg[tower_id] = {
                "correct_programs": [program],
            }
            continue

        entry = towers_cfg[tower_id]
        if program not in entry["correct_programs"]:
            entry["correct_programs"].append(program)

    return towers_cfg


def get_correct_programs(towers_cfg: Dict[str, Dict], tower_id: str) -> List[str]:
    info = towers_cfg[tower_id]
    return info["correct_programs"]


def program_to_actions(program: str) -> List[str]:
    return [tok for tok in program.strip().split() if tok]
