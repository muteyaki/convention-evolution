import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent import DyadAgent
from population import GameLogEntry, run_dyad, run_population


@dataclass
class LexiconEntry:
    meaning: str
    program: str
    utterance: str


# ---------------------------
# Lexicon utilities
# ---------------------------

def load_lexicon_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_meaning_prior_from_config(config: Dict[str, Any], lam: float = 0.8) -> Dict[str, float]:
    """P(m) ∝ exp(-lam * (L(m) - 1)), where L is program length."""
    weights: Dict[str, float] = {}
    for meaning in config.get("meanings", []):
        name = meaning["name"]
        length = meaning.get("length")
        if length is None:
            program = meaning.get("program", "")
            length = len(program.split()) if program else 1
        weights[name] = math.exp(-lam * (length - 1))

    Z = sum(weights.values())
    if Z <= 0:
        return {}
    return {name: w / Z for name, w in weights.items()}


def counts_from_prior(
    meaning_prior: Dict[str, float],
    total: int = 200,
    min_count: int = 1,
) -> Dict[str, int]:
    """Turn a probability prior into integer counts for sampling."""
    counts = {}
    for name, p in meaning_prior.items():
        counts[name] = max(min_count, int(round(p * total)))
    return counts


def sample_initial_lexicon(
    config: Dict[str, Any],
    counts: Dict[str, int],
    seed: Optional[int] = None,
) -> List[LexiconEntry]:
    """Sample utterances per meaning according to counts and templates."""
    rng = random.Random(seed)
    entries: List[LexiconEntry] = []

    for meaning in config.get("meanings", []):
        name = meaning["name"]
        kind = meaning.get("kind", "primitive")
        program = meaning.get("program", "")
        templates = meaning.get("templates", [])

        n = counts.get(name, 0)
        if n <= 0:
            continue

        if kind == "primitive":
            if not templates:
                continue
            utterance = templates[0]
            for _ in range(n):
                entries.append(LexiconEntry(meaning=name, program=program, utterance=utterance))
        else:
            if not templates:
                continue
            if n <= len(templates):
                chosen = rng.sample(templates, n)
            else:
                chosen = [rng.choice(templates) for _ in range(n)]
            for u in chosen:
                entries.append(LexiconEntry(meaning=name, program=program, utterance=u))

    return entries


def load_initial_lexicon(path: Path) -> List[LexiconEntry]:
    """Load JSONL lexicon entries."""
    entries: List[LexiconEntry] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            entries.append(
                LexiconEntry(
                    meaning=obj["meaning"],
                    program=obj["program"],
                    utterance=obj["utterance"],
                )
            )
    return entries


def save_lexicon_jsonl(path: Path, lexicon_entries: Iterable[LexiconEntry]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for entry in lexicon_entries:
            f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")


# ---------------------------
# Convenience helpers
# ---------------------------

def extract_meanings(entries: Iterable[LexiconEntry]) -> List[str]:
    return sorted({e.meaning for e in entries})


def extract_utterances(entries: Iterable[LexiconEntry]) -> List[str]:
    return sorted({e.utterance for e in entries})


def run_dyad_from_entries(
    entries: List[LexiconEntry],
    meaning_prior: Dict[str, float],
    n_rounds: int = 200,
    speaker_mode: str = "pragmatic",
    listener_mode: str = "pragmatic",
    seed: int = 0,
) -> Tuple[List[GameLogEntry], float]:
    meanings = extract_meanings(entries)
    utterances = extract_utterances(entries)
    agent_a = DyadAgent(meanings, utterances, meaning_prior, seed=seed)
    agent_b = DyadAgent(meanings, utterances, meaning_prior, seed=seed + 1)
    logs = run_dyad(
        agent_a,
        agent_b,
        meanings=meanings,
        n_rounds=n_rounds,
        speaker_mode=speaker_mode,
        listener_mode=listener_mode,
        seed=seed,
    )
    accuracy = sum(1 for x in logs if x.success) / len(logs) if logs else 0.0
    return logs, accuracy


def run_population_from_entries(
    n_dyads: int,
    entries: List[LexiconEntry],
    meaning_prior: Dict[str, float],
    n_rounds_per_dyad: int = 200,
    speaker_mode: str = "pragmatic",
    listener_mode: str = "pragmatic",
    seed: int = 0,
):
    return run_population(
        n_dyads=n_dyads,
        lexicon_entries=entries,
        meaning_prior=meaning_prior,
        n_rounds_per_dyad=n_rounds_per_dyad,
        speaker_mode=speaker_mode,
        listener_mode=listener_mode,
        seed=seed,
    )


if __name__ == "__main__":
    config_path = Path("data/task/lexicon.json")
    config = load_lexicon_config(config_path)
    meaning_prior = compute_meaning_prior_from_config(config, lam=0.8)
    counts = counts_from_prior(meaning_prior, total=200, min_count=1)
    lexicon_entries = sample_initial_lexicon(config, counts, seed=0)

    logs, acc = run_dyad_from_entries(
        lexicon_entries,
        meaning_prior,
        n_rounds=50,
        speaker_mode="pragmatic",
        listener_mode="pragmatic",
        seed=42,
    )

    print(f"Demo dyad finished. rounds={len(logs)}, accuracy={acc:.3f}")
