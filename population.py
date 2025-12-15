from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents import AgentConfig, DyadAgent
from config import *
from lexicon import build_entries_with_prior, load_lexicon_config
from task import load_towers_config, program_to_actions


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    Z = sum(dist.values())
    if Z <= 0:
        return {k: 0.0 for k in dist}
    return {k: v / Z for k, v in dist.items()}


def _snapshot_agent(agent: DyadAgent) -> Dict[str, Any]:
    """Lightweight normalized belief snapshot for inspection."""
    return {
        "role": agent.role,
        "meaning_to_utterance": {m: _normalize(agent.belief_m2u.get(m, {})) for m in agent.meanings},
        "task_to_program": {task: _normalize(probs) for task, probs in agent.task_program_belief.items()},
    }


@dataclass
class Dyad:
    """Container for a speaker–listener pair with stable roles."""

    dyad_id: int
    speaker: DyadAgent
    listener: DyadAgent

    @property
    def agents(self) -> Tuple[DyadAgent, DyadAgent]:
        return self.speaker, self.listener


def sample_dyads(
    n_dyads: int,
    entries: Sequence[Dict[str, Any]],
    lexicon_prior: Dict[str, Dict[str, float]],
    meaning_prior: Dict[str, float],
    towers_cfg: Dict[str, Dict[str, Any]],
    *,
    seed: int = GLOBAL_SEED,
    speaker_role: str = "pragmatic",
    listener_role: str = "pragmatic",
    log_dir: Optional[str] = None,
) -> List[Dyad]:
    """
    Sample {n_dyads} DyadAgent pairs (speaker + listener).
    Seeds are offset per agent to keep independent randomness.
    """
    rng = random.Random(seed)
    log_root = Path(log_dir) if log_dir else None
    if log_root:
        log_root.mkdir(parents=True, exist_ok=True)

    dyads: List[Dyad] = []
    for i in range(n_dyads):
        speaker_cfg = AgentConfig(seed=seed + 2 * i + 1)
        listener_cfg = AgentConfig(seed=seed + 2 * i + 2)

        speaker_log = str(log_root / f"dyad_{i:03d}_speaker.json") if log_root else None
        listener_log = str(log_root / f"dyad_{i:03d}_listener.json") if log_root else None

        speaker = DyadAgent(
            lexicon_entries=entries,
            lexicon_prior=lexicon_prior,
            meaning_prior=meaning_prior,
            towers_cfg=towers_cfg,
            cfg=speaker_cfg,
            role=speaker_role,
            log_path=speaker_log,
        )
        listener = DyadAgent(
            lexicon_entries=entries,
            lexicon_prior=lexicon_prior,
            meaning_prior=meaning_prior,
            towers_cfg=towers_cfg,
            cfg=listener_cfg,
            role=listener_role,
            log_path=listener_log,
        )

        dyads.append(Dyad(dyad_id=i, speaker=speaker, listener=listener))
        rng.random()  # advance rng a bit to keep parity with seed usage

    return dyads


def pack_population(dyads: Sequence[Dyad]) -> List[DyadAgent]:
    """
    Flatten dyads into a population list while keeping each agent's role attribute.
    Useful for metrics like convention strength.
    """
    population: List[DyadAgent] = []
    for dyad in dyads:
        population.extend([dyad.speaker, dyad.listener])
    return population


class Population:
    """
    Population-level trainer.
    Learn across dyads for several generations, then evaluate per-dyad success/loss
    and population convention strength.
    """

    def __init__(
        self,
        dyads: Sequence[Dyad],
        towers_cfg: Dict[str, Dict[str, Any]],
        lexicon_entries: Sequence[Dict[str, Any]],
        *,
        rounds_per_dyad: int = ROUNDS_PER_DYAD,
        seed: int = GLOBAL_SEED,
        training_mode: str = "paired",
    ) -> None:
        if not towers_cfg:
            raise ValueError("No tasks/towers loaded for population training.")
        self.dyads: List[Dyad] = list(dyads)
        self.towers_cfg = towers_cfg
        self.tower_ids = list(towers_cfg.keys())
        self.lexicon_entries = list(lexicon_entries)
        self.rounds_per_dyad = rounds_per_dyad
        self.rng = random.Random(seed)
        self.history: List[Dict[str, Any]] = []
        self.meanings = sorted({e["meaning"] for e in self.lexicon_entries})
        self.training_mode = training_mode

    # ------------- core mechanics -------------

    @property
    def agents(self) -> List[DyadAgent]:
        return pack_population(self.dyads)

    def _task_schedule(self, total_rounds: int) -> List[str]:
        cycles = math.ceil(total_rounds / len(self.tower_ids))
        schedule: List[str] = []
        for _ in range(cycles):
            shuffled = self.tower_ids[:]
            self.rng.shuffle(shuffled)
            schedule.extend(shuffled)
        return schedule[:total_rounds]

    def _compute_loss_and_success(
        self, sampled_program: str, decoded_tokens: List[str], target_tokens: List[str]
    ) -> Tuple[int, float, float, str]:
        guess_program = " ".join(decoded_tokens) if decoded_tokens else ""
        success = 1 if guess_program == sampled_program else 0

        min_len = min(len(target_tokens), len(decoded_tokens))
        mismatches = sum(1 for a, b in zip(target_tokens[:min_len], decoded_tokens[:min_len]) if a != b)
        length_gap = abs(len(target_tokens) - len(decoded_tokens))
        raw_loss = mismatches + length_gap

        denom = len(target_tokens) if target_tokens else 1
        loss = raw_loss / denom
        step_acc = 1.0 - (mismatches / denom) if denom else 0.0
        return success, loss, step_acc, guess_program

    def _run_round(
        self,
        speaker: DyadAgent,
        listener: DyadAgent,
        tower_id: str,
        round_idx: int,
        generation_idx: int,
    ) -> Dict[str, Any]:
        sampled_program, utterance_seq = speaker.produce_message_for_task(tower_id)
        target_tokens = program_to_actions(sampled_program)
        decoded_tokens = listener.interpret_message(utterance_seq)

        success, loss, step_acc, guess_program = self._compute_loss_and_success(
            sampled_program, decoded_tokens, target_tokens
        )

        speaker.observe_interaction(tower_id, sampled_program, utterance_seq, decoded_tokens)
        listener.observe_interaction(tower_id, sampled_program, utterance_seq, decoded_tokens)

        return {
            "generation": generation_idx,
            "round": round_idx,
            "tower_id": tower_id,
            "target_program": sampled_program,
            "utterance_seq": utterance_seq,
            "decoded_program": guess_program,
            "success": success,
            "loss": loss,
            "step_acc": step_acc,
        }

    # ------------- learning + eval -------------

    def learn_generation(
        self,
        generation_idx: int,
        rounds_per_dyad: Optional[int] = None,
        training_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run one generation of training.

        training_mode:
          - "paired": each dyad trains within itself (default, original behavior).
          - "mixed": speaker comes from any dyad, listener from any dyad (cross-dyad).
        """
        mode = (training_mode or self.training_mode).lower()
        n_rounds = rounds_per_dyad or self.rounds_per_dyad
        records: List[Dict[str, Any]] = []

        if mode not in {"paired", "mixed"}:
            raise ValueError(f"Unknown training_mode '{mode}', expected 'paired' or 'mixed'.")

        if mode == "paired":
            schedule = self._task_schedule(n_rounds)
            for dyad in self.dyads:
                for round_idx, tower_id in enumerate(schedule):
                    record = self._run_round(dyad.speaker, dyad.listener, tower_id, round_idx, generation_idx)
                    record.update({"speaker_dyad_id": dyad.dyad_id, "listener_dyad_id": dyad.dyad_id})
                    records.append(record)
        else:  # mixed
            # ensure each dyad serves as speaker/listener equally often: R blocks, each block shuffles all dyads
            total_rounds = n_rounds * len(self.dyads)
            schedule = self._task_schedule(total_rounds)

            speaker_sequence: List[int] = []
            listener_sequence: List[int] = []
            dyad_indices = list(range(len(self.dyads)))
            for _ in range(n_rounds):
                self.rng.shuffle(dyad_indices)
                speaker_sequence.extend(dyad_indices)
                self.rng.shuffle(dyad_indices)
                listener_sequence.extend(dyad_indices)

            for round_idx, tower_id in enumerate(schedule):
                s_idx = speaker_sequence[round_idx]
                l_idx = listener_sequence[round_idx]
                dyad_s = self.dyads[s_idx]
                dyad_l = self.dyads[l_idx]
                record = self._run_round(dyad_s.speaker, dyad_l.listener, tower_id, round_idx, generation_idx)
                record.update({"speaker_dyad_id": dyad_s.dyad_id, "listener_dyad_id": dyad_l.dyad_id})
                records.append(record)

        return records

    def evaluate_generation(self, generation_idx: int, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-dyad success/loss and population convention strength."""
        dyad_summaries: List[Dict[str, Any]] = []
        for dyad in self.dyads:
            speaker_recs = [r for r in records if r["speaker_dyad_id"] == dyad.dyad_id]
            listener_recs = [r for r in records if r["listener_dyad_id"] == dyad.dyad_id]
            successes = [r["success"] for r in speaker_recs]
            losses = [r["loss"] for r in speaker_recs]
            dyad_summaries.append(
                {
                    "dyad_id": dyad.dyad_id,
                    "speaker_role": dyad.speaker.role,
                    "listener_role": dyad.listener.role,
                    "accuracy": sum(successes) / len(successes) if successes else 0.0,
                    "avg_loss": sum(losses) / len(losses) if losses else 0.0,
                    "records": speaker_recs,  # keep primary view: how this dyad's speaker performs
                    "listener_records": listener_recs,
                }
            )

        generation_result = {
            "generation": generation_idx,
            "records": records,
            "dyads": dyad_summaries,
            "convention_strength": self.convention_strength(),
            "mean_accuracy": sum(d["accuracy"] for d in dyad_summaries) / len(dyad_summaries)
            if dyad_summaries
            else 0.0,
        }
        self.history.append(generation_result)
        return generation_result

    def run_generation(
        self,
        generation_idx: int,
        rounds_per_dyad: Optional[int] = None,
        training_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        learned = self.learn_generation(
            generation_idx,
            rounds_per_dyad=rounds_per_dyad,
            training_mode=training_mode,
        )
        return self.evaluate_generation(generation_idx, learned)

    def run(self, n_generations: int, training_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        for g in range(n_generations):
            self.run_generation(g, training_mode=training_mode)
        return self.history

    # ------------- metrics -------------

    def convention_strength(self) -> float:
        """Agreement over preferred utterances for each meaning across all agents."""
        if not self.meanings:
            return 0.0
        agreements: List[float] = []
        for meaning in self.meanings:
            votes: List[str] = []
            for agent in self.agents:
                dist = agent.belief_m2u.get(meaning, {})
                if not dist:
                    continue
                utterance_star = max(dist.items(), key=lambda kv: kv[1])[0]
                votes.append(utterance_star)
            if not votes:
                continue
            counts = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            agreements.append(max(counts.values()) / len(votes))
        return sum(agreements) / len(agreements) if agreements else 0.0


def build_population(
    dyads: Sequence[Dyad],
    towers_cfg: Dict[str, Dict[str, Any]],
    lexicon_entries: Sequence[Dict[str, Any]],
    *,
    rounds_per_dyad: int = ROUNDS_PER_DYAD,
    seed: int = GLOBAL_SEED,
    training_mode: str = "paired",
) -> Population:
    """Helper to pack dyads into a Population while preserving speaker/listener roles."""
    return Population(
        dyads=dyads,
        towers_cfg=towers_cfg,
        lexicon_entries=lexicon_entries,
        rounds_per_dyad=rounds_per_dyad,
        seed=seed,
        training_mode=training_mode,
    )


def run_population(
    *,
    n_dyads: int,
    entries: Sequence[Dict[str, Any]],
    lexicon_prior: Dict[str, Dict[str, float]],
    meaning_prior: Dict[str, float],
    towers_cfg: Dict[str, Dict[str, Any]],
    n_rounds_per_dyad: int = ROUNDS_PER_DYAD,
    n_generations: int = 1,
    seed: int = GLOBAL_SEED,
    speaker_role: str = "pragmatic",
    listener_role: str = "pragmatic",
    log_dir: Optional[str] = None,
    results_dir: str = "results",
    training_mode: str = "paired",
) -> Dict[str, Any]:
    """
    Convenience wrapper:
    1) sample dyads,
    2) build a population,
    3) train/evaluate for n_generations.
    """
    dyads = sample_dyads(
        n_dyads=n_dyads,
        entries=entries,
        lexicon_prior=lexicon_prior,
        meaning_prior=meaning_prior,
        towers_cfg=towers_cfg,
        seed=seed,
        speaker_role=speaker_role,
        listener_role=listener_role,
        log_dir=log_dir,
    )
    population = build_population(
        dyads=dyads,
        towers_cfg=towers_cfg,
        lexicon_entries=entries,
        rounds_per_dyad=n_rounds_per_dyad,
        seed=seed,
        training_mode=training_mode,
    )
    generations = population.run(n_generations=n_generations, training_mode=training_mode)
    latest = generations[-1] if generations else {"dyads": [], "convention_strength": 0.0}

    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    results_obj = {
        "generations": generations,
        "population": latest.get("dyads", []),
        "convention_strength": latest.get("convention_strength", 0.0),
    }

    with (results_dir_path / "population_results.json").open("w", encoding="utf-8") as f:
        json.dump(results_obj, f, ensure_ascii=False, indent=2)

    dyad_beliefs = [
        {
            "dyad_id": dyad.dyad_id,
            "speaker": _snapshot_agent(dyad.speaker),
            "listener": _snapshot_agent(dyad.listener),
        }
        for dyad in dyads
    ]
    with (results_dir_path / "dyad_beliefs.json").open("w", encoding="utf-8") as f:
        json.dump(dyad_beliefs, f, ensure_ascii=False, indent=2)

    return results_obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Population evolution demo based on DyadAgents.")
    parser.add_argument("--dyads", type=int, default=N_DYADS, help="Number of dyads (speaker-listener pairs).")
    parser.add_argument(
        "--rounds-per-dyad",
        type=int,
        default=ROUNDS_PER_DYAD,
        help="Training rounds per dyad in each generation.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="How many generations to simulate for population evolution.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paired",
        choices=["paired", "mixed"],
        help="Training mode: paired (each dyad self-trains) or mixed (cross-dyad speaker/listener).",
    )
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed for reproducibility.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional directory to store per-agent sampling logs.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to store generation summaries and dyad belief snapshots.",
    )
    args = parser.parse_args()

    lex_cfg = load_lexicon_config(LEXICON_CONFIG_PATH)
    entries, lexicon_prior, meaning_prior = build_entries_with_prior(lex_cfg, lam=LENGTH_PRIOR_LAMBDA)
    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)

    results = run_population(
        n_dyads=args.dyads,
        entries=entries,
        lexicon_prior=lexicon_prior,
        meaning_prior=meaning_prior,
        towers_cfg=towers_cfg,
        n_rounds_per_dyad=args.rounds_per_dyad,
        n_generations=args.generations,
        seed=args.seed,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        training_mode=args.mode,
    )

    results_dir_path = Path(args.results_dir)
    print(f"Saved population summary to {results_dir_path / 'population_results.json'}")
    print(f"Saved dyad belief snapshots to {results_dir_path / 'dyad_beliefs.json'}")

    last_gen = results["generations"][-1] if results["generations"] else {}
    conv = last_gen.get("convention_strength", 0.0)
    mean_acc = last_gen.get("mean_accuracy", 0.0)
    print(f"Simulated {args.generations} generations × {args.dyads} dyads.")
    print(f"Final generation mean accuracy: {mean_acc:.3f}, convention strength: {conv:.3f}")
    for dyad in last_gen.get("dyads", []):
        print(
            f"  Dyad {dyad['dyad_id']:02d} "
            f"({dyad['speaker_role']}→{dyad['listener_role']}): "
            f"acc={dyad['accuracy']:.3f}, loss={dyad['avg_loss']:.3f}"
        )


if __name__ == "__main__":
    main()
