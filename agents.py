import random
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from config import *
from task import program_to_actions, program_length
from lexicon import get_utterance_action_length,get_meaning_for_utterance


@dataclass
class AgentConfig:
    alpha_speaker: float = ALPHA_SPEAKER
    beta_t: float = TASK_COST_WEIGHT
    epsilon: float = EPSILON
    seed: Optional[int] = GLOBAL_SEED
    update_weight: float = BELIF_UPDATE


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

        # print(f"Initialized agent belief_m2u: {self.belief_m2u}")
        # print(f"Initialized agent belief_u2m: {self.belief_u2m}")

        # task→program belief (length prior)
        self.task_program_belief: Dict[str, Dict[str, float]] = {}
        for task_id, info in towers_cfg.items():
            programs = info["correct_programs"]
            weights = {p: math.exp(LENGTH_PRIOR_LAMBDA * (program_length(p) - 1)) for p in programs}
            self.task_program_belief[task_id] = _normalize(weights)
        # print(f"Initialized agent task_program_belief: {self.task_program_belief}")

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
        dist = _normalize(exps)

        programs = list(dist.keys())
        probs = [dist[p] for p in programs]
        choice = random.choices(programs, weights=probs, k=1)[0]
        # choice = max(dist.items(), key=lambda kv: kv[1])[0]
        self._log_sample(
            {
                "event": "sample_program_for_task",
                "task_id": task_id,
                "dist": dist,
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
        dist = self.speaker_distribution(meaning)
        if not dist:
            raise ValueError(f"No utterances available for meaning in speaker belief: {meaning}")
        utts = list(dist.keys())
        probs = [dist[u] for u in utts]
        choice = random.choices(utts, weights=probs, k=1)[0]
        self._log_sample(
            {
                "event": "sample_utterance_for_meaning",
                "meaning": meaning,
                "dist": dist,
                "choice": choice,
            }
        )
        return choice

    def interpret_utterance(self, utterance: str) -> str:
        dist = self.literal_listener(utterance) if self.role == "literal" else self.pragmatic_listener(utterance)
        if not dist:
            raise ValueError(f"No meanings available for utterance in listener belief: {utterance}")
        # return max(dist.items(), key=lambda kv: kv[1])[0]
        ms = list(dist.keys())
        ps = [dist[m] for m in ms]
        choice = random.choices(ms, weights=ps, k=1)[0]
        self._log_sample(
            {
                "event": "sample_meaning_for_utterance",
                "utterance": utterance,
                "listener_mode": self.role,
                "dist": dist,
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
        scores: Dict[str, float] = {}
        for u in self.utterances:
            L0 = self.literal_listener(u)
            p = L0.get(meaning, self.cfg.epsilon)
            utility = math.log(p + self.cfg.epsilon)
            scores[u] = self.cfg.alpha_speaker * utility
        # softmax
        max_score = max(scores.values())
        exps = {u: math.exp(s - max_score) for u, s in scores.items()}
        return _normalize(exps)

    # Pragmatic listener L1(m|u)
    def pragmatic_listener(self, utterance: str) -> Dict[str, float]:
        post: Dict[str, float] = {}
        for m in self.meanings:
            S1 = self.speaker_distribution(m).get(utterance, self.cfg.epsilon)
            prior = self.lexicon_prior[m][utterance]
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
