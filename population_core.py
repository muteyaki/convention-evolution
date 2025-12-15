from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agents import AgentConfig, DyadAgent
from config import *
from task import program_to_actions


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    Z = sum(dist.values())
    if Z <= 0:
        return {k: 0.0 for k in dist}
    return {k: v / Z for k, v in dist.items()}


def _normalize_with_support(dist: Dict[str, float], support: Sequence[str]) -> Dict[str, float]:
    if not support:
        return {}
    restricted = {k: dist.get(k, 0.0) for k in support}
    total = sum(restricted.values())
    if total <= 0:
        uniform = 1.0 / len(support)
        return {k: uniform for k in support}
    return {k: v / total for k, v in restricted.items()}


def _population_mean_distribution(dists: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not dists:
        return {}
    keys = set()
    for dist in dists:
        keys.update(dist.keys())
    if not keys:
        return {}
    mean = {k: 0.0 for k in keys}
    for dist in dists:
        for k in keys:
            mean[k] += dist.get(k, 0.0)
    n = len(dists)
    if n <= 0:
        return {}
    mean = {k: v / n for k, v in mean.items()}
    return _normalize(mean)


def _kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    total = 0.0
    for k, p_val in p.items():
        if p_val <= 0:
            continue
        q_val = max(q.get(k, 0.0), eps)
        total += p_val * math.log(p_val / q_val)
    return total


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    p_full = {k: p.get(k, 0.0) for k in keys}
    q_full = {k: q.get(k, 0.0) for k in keys}
    m = {k: 0.5 * (p_full[k] + q_full[k]) for k in keys}
    m = _normalize(m)
    return 0.5 * _kl_divergence(p_full, m) + 0.5 * _kl_divergence(q_full, m)


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
        self.utterances = sorted({e["utterance"] for e in self.lexicon_entries})
        self.program_support: Dict[str, List[str]] = {
            task_id: list(info.get("correct_programs", [])) for task_id, info in self.towers_cfg.items()
        }
        self.training_mode = training_mode
        self._last_round_belief_history: List[Dict[str, Any]] = []

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
        self, sampled_program: str, decoded_actions: List[str], target_actions: List[str]
    ) -> Tuple[int, float, float, str]:
        guess_program = " ".join(decoded_actions) if decoded_actions else ""
        success = 1 if guess_program == sampled_program else 0

        decoded_len = len(decoded_actions)
        target_len = len(target_actions)
        overlap = min(target_len, decoded_len)
        mismatches = sum(1 for a, b in zip(target_actions[:overlap], decoded_actions[:overlap]) if a != b)
        length_diff = abs(target_len - decoded_len)
        denom = target_len if target_len > 0 else 1
        loss = (mismatches + length_diff) / denom
        step_acc = (overlap - mismatches) / overlap if overlap > 0 else 0.0
        return success, loss, step_acc, guess_program

    def _run_round(
        self,
        speaker: DyadAgent,
        listener: DyadAgent,
        tower_id: str,
        round_idx: int,
    ) -> Dict[str, Any]:
        sampled_program, utterance_seq = speaker.produce_message_for_task(tower_id)
        target_actions = program_to_actions(sampled_program)
        decoded_actions = listener.interpret_message(utterance_seq)

        success, loss, step_acc, guess_program = self._compute_loss_and_success(
            sampled_program, decoded_actions , target_actions
        )

        speaker.observe_interaction(tower_id, sampled_program, utterance_seq, decoded_actions)
        listener.observe_interaction(tower_id, sampled_program, utterance_seq, decoded_actions)

        return {
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
        round_history: List[Dict[str, Any]] = []
        round_counter = 0

        if mode not in {"paired", "mixed"}:
            raise ValueError(f"Unknown training_mode '{mode}', expected 'paired' or 'mixed'.")

        def _record_mean(round_idx: int) -> None:
            round_history.append(
                {
                    "round": round_idx,
                    "mode": mode,
                    "evolute_convention": self.evolute_convention(),
                }
            )

        if mode == "paired":
            schedule = self._task_schedule(n_rounds)
            for dyad in self.dyads:
                for round_idx, tower_id in enumerate(schedule):
                    record = self._run_round(dyad.speaker, dyad.listener, tower_id, round_idx)
                    record.update({"speaker_dyad_id": dyad.dyad_id, "listener_dyad_id": dyad.dyad_id})
                    records.append(record)
                    _record_mean(round_counter)
                    round_counter += 1
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
                record = self._run_round(dyad_s.speaker, dyad_l.listener, tower_id, round_idx)
                record.update({"speaker_dyad_id": dyad_s.dyad_id, "listener_dyad_id": dyad_l.dyad_id})
                records.append(record)
                _record_mean(round_counter)
                round_counter += 1

        self._last_round_belief_history = round_history
        return records

    def evaluate_generation(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-dyad success/loss and population-level belief stats."""
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
            "records": records,
            "dyads": dyad_summaries,
            "evolute_convention": self.evolute_convention(),
            "mean_belief_history": list(self._last_round_belief_history),
            "mean_accuracy": sum(d["accuracy"] for d in dyad_summaries) / len(dyad_summaries)
            if dyad_summaries
            else 0.0,
        }
        self.history.append(generation_result)
        return generation_result

    def run_generation(
        self,
        rounds_per_dyad: Optional[int] = None,
        training_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        learned = self.learn_generation(
            rounds_per_dyad=rounds_per_dyad,
            training_mode=training_mode,
        )
        return self.evaluate_generation(learned)

    def run(self, n_generations: int, training_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        for g in range(n_generations):
            self.run_generation(g, training_mode=training_mode)
        return self.history

    # ------------- metrics -------------

    def _aggregate_belief_stats(
        self,
        keys: Sequence[str],
        dist_getter: Callable[[DyadAgent, str], Dict[str, float]],
        support_getter: Optional[Callable[[str], Sequence[str]]] = None,
        *,
        agents: Optional[Sequence[DyadAgent]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[float]]:
        """Compute per-key population means and Jensen–Shannon divergence values."""
        agents_list = list(agents) if agents is not None else self.agents
        means: Dict[str, Dict[str, float]] = {}
        js_values: List[float] = []
        for key in keys:
            support = list(support_getter(key)) if support_getter else None
            if support is not None and not support:
                continue
            dists: List[Dict[str, float]] = []
            for agent in agents_list:
                raw = dist_getter(agent, key)
                if support is not None:
                    dist = _normalize_with_support(raw, support)
                else:
                    dist = _normalize(raw)
                if dist:
                    dists.append(dist)
            if not dists:
                continue
            mean = _population_mean_distribution(dists)
            means[key] = mean
            for dist in dists:
                js_values.append(_js_divergence(dist, mean))
        return means, js_values

    def evolute_convention(self) -> Dict[str, Any]:
        """Population belief alignment stats based on Jensen–Shannon divergence."""
        speaker_agents = [dyad.speaker for dyad in self.dyads]
        listener_agents = [dyad.listener for dyad in self.dyads]

        meaning_means, meaning_js = self._aggregate_belief_stats(
            self.meanings,
            lambda agent, meaning: agent.belief_m2u.get(meaning, {}),
            lambda _meaning: self.utterances,
            agents=speaker_agents,
        )
        utterance_means, utterance_js = self._aggregate_belief_stats(
            self.utterances,
            lambda agent, utterance: agent.belief_u2m.get(utterance, {}),
            lambda _utterance: self.meanings,
            agents=listener_agents,
        )
        task_means, task_js = self._aggregate_belief_stats(
            self.tower_ids,
            lambda agent, task_id: agent.task_program_belief.get(task_id, {}),
            lambda task_id: self.program_support.get(task_id, []),
            agents=speaker_agents,
        )

        all_js = meaning_js + utterance_js + task_js
        overall_js = sum(all_js) / len(all_js) if all_js else 0.0

        def _avg(js_vals: Sequence[float]) -> float:
            return sum(js_vals) / len(js_vals) if js_vals else 0.0

        return {
            "task_to_program": {"mean": task_means, "avg_js_divergence": _avg(task_js)},
            "meaning_to_utterance": {"mean": meaning_means, "avg_js_divergence": _avg(meaning_js)},
            "utterance_to_meaning": {"mean": utterance_means, "avg_js_divergence": _avg(utterance_js)},
            "overall_js_divergence": overall_js,
        }


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
    results_dir: str = RESULTS_DIR,
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
    latest = generations[-1] if generations else {"dyads": [], "evolute_convention": {}}

    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    results_obj = {
        "generations": generations,
        "population": latest.get("dyads", []),
        "evolute_convention": latest.get("evolute_convention", {}),
        "mean_belief_history": latest.get("mean_belief_history", []),
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
