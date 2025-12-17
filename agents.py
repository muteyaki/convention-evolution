"""RSA-style dyad agents and belief updates."""

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ALPHA_SPEAKER,
    BELIF_UPDATE,
    CHOICE_TEMPERATURE,
    EPSILON,
    GLOBAL_SEED,
    LENGTH_PRIOR_LAMBDA,
    TASK_COST_WEIGHT,
)
from lexicon import get_meaning_for_utterance
from task import program_length, program_to_actions


@dataclass
class AgentConfig:
    alpha_speaker: float = ALPHA_SPEAKER
    beta_t: float = TASK_COST_WEIGHT
    epsilon: float = EPSILON
    seed: Optional[int] = GLOBAL_SEED
    update_weight: float = BELIF_UPDATE
    choice_temperature: float = CHOICE_TEMPERATURE


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    Z = sum(dist.values())
    if Z <= 0:
        uniform = 1.0 / len(dist) if dist else 0.0
        return {k: uniform for k in dist}
    return {k: v / Z for k, v in dist.items()}


class DyadAgent:
    """Agent with task→program belief and meaning↔utterance beliefs."""

    def __init__(
        self,
        lexicon_entries: List[Dict[str, Any]],
        lexicon_prior: Dict[str, Dict[str, float]],
        meaning_prior: Dict[str, float],
        towers_cfg: Dict[str, Dict],
        cfg: AgentConfig,
        role: str = "pragmatic",
        log_path: Optional[str] = None,
    ):
        if cfg.seed is not None:
            random.seed(cfg.seed)
        self.cfg = cfg
        self.role = role
        self.meanings = sorted({e["meaning"] for e in lexicon_entries})
        self.utterances = sorted({e["utterance"] for e in lexicon_entries})
        self.meaning_prior = meaning_prior
        self.lexicon_entries = list(lexicon_entries)
        self.lexicon_prior: Dict[str, Dict[str, float]] = {
            m: dict(u_dict) for m, u_dict in lexicon_prior.items()
        }

        # meaning↔utterance beliefs (initialized from merged prior: length-based freq + fidelity boost on aligned pairs)
        self.belief_m2u: Dict[str, Dict[str, float]] = {
            m: {u: self.lexicon_prior.get(m, {}).get(u, 0.0) for u in self.utterances}
            for m in self.meanings
        }
        self.belief_u2m: Dict[str, Dict[str, float]] = {
            u: {m: self.belief_m2u[m].get(u, 0.0) for m in self.meanings} for u in self.utterances
        }

        # task→program belief (length prior)
        self.task_program_belief: Dict[str, Dict[str, float]] = {}
        for task_id, info in towers_cfg.items():
            programs = info["correct_programs"]
            weights = {p: math.exp(LENGTH_PRIOR_LAMBDA * (program_length(p) - 1)) for p in programs}
            self.task_program_belief[task_id] = _normalize(weights)

        # logging
        self.log_path = log_path
        self._log_records: List[Dict[str, Any]] = []

    def set_log_path(self, path: str) -> None:
        """Enable logging sampled distributions to a JSON file (overwrites on each write)."""
        self.log_path = path
        self._log_records = []
        self._flush_logs()

    def _flush_logs(self) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._log_records, f, ensure_ascii=False, indent=2)

    def _log_sample(self, entry: Dict[str, Any]) -> None:
        """Append a log entry and persist to disk if a log path is configured."""
        entry["agent_role"] = self.role
        self._log_records.append(entry)
        self._flush_logs()

    def _apply_temperature(self, dist: Dict[str, float]) -> Dict[str, float]:
        """Adjust distribution sharpness via τ; τ→0 -> argmax, τ large -> uniform."""
        if not dist:
            return {}
        tau = max(self.cfg.choice_temperature, self.cfg.epsilon)
        if abs(tau - 1.0) < 1e-9:
            return dict(dist)
        adjusted = {k: math.pow(max(v, self.cfg.epsilon), 1.0 / tau) for k, v in dist.items()}
        return _normalize(adjusted)

    def _sample_from_dist(self, dist: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """Sample a key from a probability dict after temperature adjustment."""
        adjusted = self._apply_temperature(dist)
        if not adjusted:
            raise ValueError("Cannot sample from an empty distribution.")
        keys = list(adjusted.keys())
        weights = [adjusted[k] for k in keys]
        choice = random.choices(keys, weights=weights, k=1)[0]
        return choice, adjusted

    # ---------------- Task → Program ----------------
    def sample_program_for_task(self, task_id: str) -> str:
        belief = self.task_program_belief.get(task_id, {})
        norm_belief = _normalize(belief)
        if not belief:
            raise ValueError(f"No program belief for task: {task_id}")

        scores: Dict[str, float] = {}
        for p, prob in norm_belief.items():
            cost = program_length(p)
            utility = (1 - self.cfg.beta_t) * math.log(prob + self.cfg.epsilon) - self.cfg.beta_t * math.log(cost)
            scores[p] = self.cfg.alpha_speaker * utility

        max_score = max(scores.values())
        exps = {p: math.exp(s - max_score) for p, s in scores.items()}
        base_dist = _normalize(exps)
        choice, sample_dist = self._sample_from_dist(base_dist)
        # choice = max(dist.items(), key=lambda kv: kv[1])[0]
        self._log_sample(
            {
                "event": "sample_program_for_task",
                "task_id": task_id,
                "dist": sample_dist,
                "raw_dist": base_dist,
                "choice": choice,
            }
        )
        return choice

    def update_task_belief(self, task_id: str, program: str, fraction_correct: float = 1.0) -> None:
        step = self.cfg.update_weight * fraction_correct
        self.task_program_belief.setdefault(task_id, {})
        self.task_program_belief[task_id][program] = self.task_program_belief[task_id].get(program, 0.0) + step
        self.task_program_belief[task_id] = _normalize(self.task_program_belief[task_id])

    # ---------------- Meaning ↔ Utterance ----------------
    def sample_utterance_for_meaning(self, meaning: str) -> str:
        speaker_dist = self.speaker_distribution(meaning)
        if not speaker_dist:
            raise ValueError(f"No utterances available for meaning in speaker belief: {meaning}")
        choice, sample_dist = self._sample_from_dist(speaker_dist)
        self._log_sample(
            {
                "event": "sample_utterance_for_meaning",
                "meaning": meaning,
                "dist": sample_dist,
                "raw_dist": speaker_dist,
                "choice": choice,
            }
        )
        return choice

    def interpret_utterance(self, utterance: str) -> str:
        listener_dist = (
            self.literal_listener(utterance) if self.role == "literal" else self.pragmatic_listener(utterance)
        )
        if not listener_dist:
            raise ValueError(f"No meanings available for utterance in listener belief: {utterance}")
        # return max(dist.items(), key=lambda kv: kv[1])[0]
        choice, sample_dist = self._sample_from_dist(listener_dist)
        self._log_sample(
            {
                "event": "sample_meaning_for_utterance",
                "utterance": utterance,
                "listener_mode": self.role,
                "dist": sample_dist,
                "raw_dist": listener_dist,
                "choice": choice,
            }
        )
        return choice

    def update_utterance_meaning(self, utterance: str, meaning: str) -> None:
        step = self.cfg.update_weight
        self.belief_m2u.setdefault(meaning, {})
        self.belief_u2m.setdefault(utterance, {})
        self.belief_m2u[meaning][utterance] = self.belief_m2u[meaning].get(utterance, 0.0) + step
        self.belief_u2m[utterance][meaning] = self.belief_u2m[utterance].get(meaning, 0.0) + step

    # Literal listener L0(m|u)
    def literal_listener(self, utterance: str) -> Dict[str, float]:
        dist = _normalize(self.belief_u2m.get(utterance, {}))
        if not dist:
            raise ValueError(f"No meanings available for utterance in literal listener belief: {utterance}")
        return dist

    # Pragmatic speaker S1(u|m)
    def speaker_distribution(self, meaning: str) -> Dict[str, float]:
        post: Dict[str, float] = {}
        dist = _normalize(self.belief_m2u.get(meaning, {}))
        for u in self.utterances:
            L0 = self.literal_listener(u).get(meaning, self.cfg.epsilon)
            prior = dist.get(u)
            post[u] = L0 * prior
        return _normalize(post)

    # Pragmatic listener L1(m|u)
    def pragmatic_listener(self, utterance: str) -> Dict[str, float]:
        post: Dict[str, float] = {}
        Pm = self.meaning_prior
        for m in self.meanings:
            S1 = self.speaker_distribution(m).get(utterance, self.cfg.epsilon)
            prior = Pm.get(m, self.cfg.epsilon) 
            post[m] = S1 * prior
        return _normalize(post)

    # ---------------- Message-level APIs ----------------
    def produce_message_for_task(self, task_id: str) -> Tuple[str, List[str]]:
        program = self.sample_program_for_task(task_id)
        actions = program_to_actions(program)
        utterances = [self.sample_utterance_for_meaning(m) for m in actions]
        return program, utterances

    def interpret_message(self, utterances: List[str]) -> List[str]:
        return [self.interpret_utterance(u) for u in utterances]

    def observe_interaction(
        self,
        task_id: str,
        true_program: str,
        utterances: List[str],
        decoded_tokens: Optional[List[str]] = None,
    ) -> None:
        actions = program_to_actions(true_program)
        guesses = decoded_tokens if decoded_tokens is not None else self.interpret_message(utterances)
        # Update beliefs only when got the right match
        min_len = min(len(actions), len(utterances), len(guesses))
        correct = 0
        for m_true, u, m_guess in zip(actions[:min_len], utterances[:min_len], guesses[:min_len]):
            m_gold_for_u = get_meaning_for_utterance(self.lexicon_entries, u)
            if m_true == m_guess and m_gold_for_u == m_true:
                correct += 1
                # print("we will update", m_true , "with" ,u)
                self.update_utterance_meaning(u, m_true)
        total = len(actions)
        frac = correct / total if total > 0 else 0.0
        # update task belief 
        self.update_task_belief(task_id, true_program, fraction_correct=frac)
