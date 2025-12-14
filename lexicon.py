# lexicon.py
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict
import json
import math
from config import LENGTH_PRIOR_LAMBDA, FIDELITY

from task import program_length


# -------- Read lexicon defination --------

def load_lexicon_config(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------- Build dict and prior --------

def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    Z = sum(dist.values())
    if Z <= 0:
        uniform = 1.0 / len(dist) if dist else 0.0
        return {k: uniform for k in dist}
    return {k: v / Z for k, v in dist.items()}


def build_entries_with_prior(lex_cfg: List[Dict[str, Any]], lam: float = LENGTH_PRIOR_LAMBDA) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Read from lexicon.json ：
      - entries: [{meaning, program, utterance, length}]
      - lexicon_prior: dict[meaning][utterance]，** consider the frequency of lexicons and the fidelity
    """
    entries: List[Dict[str, Any]] = []
    weights: Dict[str, float] = defaultdict(float)
    utterance_to_meaning: Dict[str, str] = {}

    for item in lex_cfg:
        meaning = item.get("meaning") or item.get("name")
        program = item["program"]
        length = item.get("length", program_length(program))
        # w = exp(-lambda * (length - 1))
        weights[meaning] += math.exp(-lam * (length - 1))
        for u in item.get("utterance", []):
            entries.append(
                {"meaning": meaning, "program": program, "utterance": u, "length": length}
            )
            utterance_to_meaning[u] = meaning

    meaning_prior = _normalize(weights)
    meanings = sorted(meaning_prior.keys())
    utterances = sorted(utterance_to_meaning.keys())

    lexicon_prior: Dict[str, Dict[str, float]] = {}
    for m in meanings:
        row: Dict[str, float] = {}
        for u in utterances:
            true_m = utterance_to_meaning[u]
            base = meaning_prior.get(true_m, 0.0)
            row[u] = base * (FIDELITY if true_m == m else 1.0)
        lexicon_prior[m] = row

    return entries, lexicon_prior


# -------- Tool functions for managing the lexicon dict --------

def get_program_for_meaning(entries: List[Dict[str, Any]], meaning: str) -> str:
    for e in entries:
        if e["meaning"] == meaning:
            return e["program"]
    raise KeyError(f"Meaning not found: {meaning}")


def get_program_length_for_meaning(entries: List[Dict[str, Any]], meaning: str) -> int:
    for e in entries:
        if e["meaning"] == meaning:
            return e.get("length", program_length(e["program"]))
    raise KeyError(f"Meaning not found: {meaning}")


def get_meaning_for_utterance(entries: List[Dict[str, Any]], utterance: str) -> str:
    for e in entries:
        if e["utterance"] == utterance:
            return e["meaning"]
    raise KeyError(f"Utterance not found: {utterance}")


def get_utterance_action_length(entries: List[Dict[str, Any]], utterance: str) -> int:
    meaning = get_meaning_for_utterance(entries, utterance)
    return get_program_length_for_meaning(entries, meaning)
