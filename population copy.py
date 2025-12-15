# population.py
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from config import * 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents import AgentConfig, DyadAgent
from task import load_towers_config, program_to_actions
from lexicon import load_lexicon_config, build_entries_with_prior


# -------------------------
# utilities
# -------------------------

def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    Z = sum(dist.values())
    if Z <= 0:
        return {k: 0.0 for k in dist}
    return {k: v / Z for k, v in dist.items()}


# -------------------------
# config
# -------------------------

@dataclass
class PopConfig:
    seed: int = GLOBAL_SEED
    n_agents: int = N_DYADS
    n_rounds: int = 200
    length_prior_lambda: float = LENGTH_PRIOR_LAMBDA

    # logging / output
    log_prefix: str = "pop"
    plots_dir: str = "plots"
    results_dir: str = "results"
    log_every: int = 200
    window: int = 500  # rolling window for success

    # turnover
    turnover_enabled: bool = True
    turnover_interval: int = ROUNDS_PER_DYAD
    turnover_fraction: float = NEWPOP_RATIO  # replace rho fraction

    # newcomer learning speed
    newcomer_tau: float = 0.8  # target success rate
    newcomer_horizon: int = ROUNDS_PER_DYAD  # track last H participations

    # if True, update beliefs only when success == 1
    update_only_on_success: bool = False


@dataclass
class AgentLife:
    born_round: int
    participations: int = 0
    successes: int = 0
    last_outcomes: List[int] = field(default_factory=list)

    def record(self, success: int, horizon: int) -> None:
        self.participations += 1
        self.successes += int(success)
        self.last_outcomes.append(int(success))
        if len(self.last_outcomes) > horizon:
            self.last_outcomes = self.last_outcomes[-horizon:]

    def rolling_sr(self) -> float:
        if not self.last_outcomes:
            return 0.0
        return sum(self.last_outcomes) / len(self.last_outcomes)

    def time_to_tau(self, tau: float) -> Optional[int]:
        # cumulative success rate threshold (simple + stable)
        if self.participations == 0:
            return None
        if (self.successes / self.participations) >= tau:
            return self.participations
        return None


# -------------------------
# population simulator
# -------------------------

class PopulationSimulator:
    def __init__(self, cfg: PopConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.plots_dir = Path(cfg.plots_dir)
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir = Path(cfg.results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.n_rounds = cfg.n_rounds

        # load configs
        self.towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)
        self.lex_cfg = load_lexicon_config(LEXICON_CONFIG_PATH)
        self.lexicon_entries, self.lexicon_prior, self.meaning_prior = build_entries_with_prior(
            self.lex_cfg, lam=cfg.length_prior_lambda
        )

        self.tower_ids = list(self.towers_cfg.keys())
        if not self.tower_ids:
            raise ValueError("No towers available in configuration.")


        # create agents
        self.agents: List[DyadAgent] = []
        self.life: Dict[int, AgentLife] = {}
        for i in range(cfg.n_agents):
            self._spawn_agent(i, born_round=0)

        # rolling window success
        self._succ_window: List[int] = []

        # logs
        self.round_records: List[Dict[str, Any]] = []
        self.snapshots: List[Dict[str, Any]] = []

        # curves
        self.losses: List[float] = []
        self.successes: List[int] = []
        self.step_accs: List[float] = []

    # -------- agent lifecycle --------

    def _spawn_agent(self, agent_id: int, born_round: int) -> None:
        # seeds per agent (like your dyad demo: cfg_A seed=seed, cfg_B seed=seed+1)
        cfg_a = AgentConfig(seed=self.cfg.seed + agent_id + 1)

        log_path = self.results_dir / f"{self.cfg.log_prefix}_agent_{agent_id:03d}.json"
        agent = DyadAgent(
            lexicon_entries=self.lexicon_entries,
            lexicon_prior=self.lexicon_prior,
            meaning_prior=self.meaning_prior,
            towers_cfg=self.towers_cfg,
            cfg=cfg_a,
            log_path=log_path,
        )

        if agent_id < len(self.agents):
            self.agents[agent_id] = agent
        else:
            self.agents.append(agent)

        self.life[agent_id] = AgentLife(born_round=born_round)

    def _turnover(self, current_round: int) -> List[int]:
        n_replace = max(1, int(self.cfg.turnover_fraction * self.cfg.n_agents))
        ids = list(range(self.cfg.n_agents))
        self.rng.shuffle(ids)
        replaced = ids[:n_replace]
        for aid in replaced:
            self._spawn_agent(aid, born_round=current_round)
        return replaced

    # -------- sampling --------

    def _sample_pair(self) -> Tuple[int, int]:
        i = self.rng.randrange(self.cfg.n_agents)
        j = self.rng.randrange(self.cfg.n_agents - 1)
        if j >= i:
            j += 1
        return i, j

    def _sample_task(self, tower_ids:list) -> str:
        cycles = math.ceil(self.n_rounds / len(tower_ids))
        tower_schedule: List[str] = []
        for _ in range(cycles):
            shuffled = tower_ids[:]
            random.shuffle(shuffled)
            tower_schedule.extend(shuffled)
        tower_schedule = tower_schedule[:self.n_rounds]
        return tower_schedule

    # -------- metrics --------

    def _compute_loss_and_success(
        self,
        sampled_program: str,
        decoded_tokens: List[str],
        target_tokens: List[str],
    ) -> Tuple[int, float, str]:
        guess_program = " ".join(decoded_tokens) if decoded_tokens else ""
        success = 1 if guess_program == sampled_program else 0

        min_len = min(len(target_tokens), len(decoded_tokens))
        mismatches = sum(
            1 for a, b in zip(target_tokens[:min_len], decoded_tokens[:min_len]) if a != b
        )
        raw_loss = mismatches

        program_len = len(target_tokens)
        denom = program_len if program_len > 0 else 1
        loss = raw_loss / denom

        return success, loss, guess_program

    def convention_strength(self) -> float:
        """Agreement over m->u preferred utterances across population."""
        meanings = sorted({e["meaning"] for e in self.lexicon_entries})
        if not meanings:
            return 0.0
        agrs = []
        for m in meanings:
            votes = []
            for a in self.agents:
                dist = a.belief_m2u.get(m, {})
                if not dist:
                    continue
                u_star = max(dist.items(), key=lambda kv: kv[1])[0]
                votes.append(u_star)
            if not votes:
                continue
            c = Counter(votes)
            agrs.append(max(c.values()) / len(votes))
        return sum(agrs) / len(agrs) if agrs else 0.0

    def avg_newcomer_time_to_tau(self, tau: float) -> Optional[float]:
        times = []
        for aid, life in self.life.items():
            t = life.time_to_tau(tau)
            if t is not None:
                times.append(t)
        if not times:
            return None
        return sum(times) / len(times)

    def rolling_success_rate(self) -> float:
        if not self._succ_window:
            return 0.0
        return sum(self._succ_window) / len(self._succ_window)

    # -------- snapshot --------

    def snapshot_agent(self, agent: DyadAgent) -> Dict[str, Dict[str, Dict[str, float]]]:
        meaning_to_utterance = {m: _normalize(agent.belief_m2u.get(m, {})) for m in agent.meanings}
        task_to_program = {task: _normalize(probs) for task, probs in agent.task_program_belief.items()}
        return {
            "meaning_to_utterance": meaning_to_utterance,
            "task_to_program": task_to_program,
        }

    # -------- main loop --------

    def run(self) -> None:
        tower_schedule = self._sample_task(self.tower_ids)
        for r in range(self.cfg.n_rounds):
            tower_id = tower_schedule[r]
            speaker_id, listener_id = self._sample_pair()

            speaker = self.agents[speaker_id]
            listener = self.agents[listener_id]

            sampled_program, utterance_seq = speaker.produce_message_for_task(tower_id)
            target_tokens = program_to_actions(sampled_program)
            decoded_tokens = listener.interpret_message(utterance_seq)

            success, loss, step_acc, guess_program = self._compute_loss_and_success(
                sampled_program, decoded_tokens, target_tokens
            )

            # update beliefs (same call style as your dyad demo)
            if (not self.cfg.update_only_on_success) or (success == 1):
                speaker.observe_interaction(tower_id, sampled_program, utterance_seq, decoded_tokens)
                listener.observe_interaction(tower_id, sampled_program, utterance_seq, decoded_tokens)

            # update histories
            self.life[speaker_id].record(success, self.cfg.newcomer_horizon)
            self.life[listener_id].record(success, self.cfg.newcomer_horizon)

            # rolling window
            self._succ_window.append(success)
            if len(self._succ_window) > self.cfg.window:
                self._succ_window = self._succ_window[-self.cfg.window :]

            # store curves
            self.losses.append(loss)
            self.successes.append(success)
            self.step_accs.append(step_acc)

            # store per-round record (lightweight)
            self.round_records.append(
                {
                    "round": r,
                    "tower_id": tower_id,
                    "speaker": speaker_id,
                    "listener": listener_id,
                    "sampled_program": sampled_program,
                    "utterance_seq": utterance_seq,
                    "guess_program": guess_program,
                    "success": success,
                    "loss": loss,
                    "step_acc": step_acc,
                    "program_len": len(target_tokens),
                    "rolling_sr": self.rolling_success_rate(),
                }
            )

            # turnover
            replaced = None
            if self.cfg.turnover_enabled and self.cfg.turnover_interval > 0 and (r + 1) % self.cfg.turnover_interval == 0:
                replaced = self._turnover(current_round=r)

            # snapshots
            if self.cfg.log_every > 0 and (r + 1) % self.cfg.log_every == 0:
                snap = {
                    "round": r,
                    "rolling_sr": self.rolling_success_rate(),
                    "agreement": self.convention_strength(),
                    "avg_newcomer_time_to_tau": self.avg_newcomer_time_to_tau(self.cfg.newcomer_tau),
                }
                if replaced:
                    snap["turnover_replaced"] = replaced
                self.snapshots.append(snap)

                print(
                    f"[round {r:05d}] sr={snap['rolling_sr']:.3f} "
                    f"agr={snap['agreement']:.3f} newcomer_T={snap['avg_newcomer_time_to_tau']}"
                )

        self._save_outputs()

    # -------- outputs --------

    def _save_outputs(self) -> None:
        # save logs
        rec_path = self.results_dir / f"{self.cfg.log_prefix}_round_records.json"
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(self.round_records, f, ensure_ascii=False, indent=2)
        print(f"Saved round records to {rec_path}")

        snap_path = self.results_dir / f"{self.cfg.log_prefix}_snapshots.json"
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(self.snapshots, f, ensure_ascii=False, indent=2)
        print(f"Saved snapshots to {snap_path}")

        # plots: loss curve
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(self.losses)), self.losses, marker="o", markersize=2, linewidth=1)
        plt.xlabel("Round")
        plt.ylabel("Loss (mismatch / length)")
        plt.title("Population Loss Curve (per round)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = self.plots_dir / f"{self.cfg.log_prefix}_loss_curve.png"
        plt.savefig(loss_path, dpi=150)
        print(f"Saved loss curve to {loss_path}")

        # plots: rolling success (use snapshots for readability)
        if self.snapshots:
            xs = [s["round"] for s in self.snapshots]
            ys = [s["rolling_sr"] for s in self.snapshots]
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, marker="o", markersize=3, linewidth=1)
            plt.xlabel("Round")
            plt.ylabel(f"Rolling SR (window={self.cfg.window})")
            plt.title("Population Success Rate (rolling)")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            sr_path = self.plots_dir / f"{self.cfg.log_prefix}_rolling_sr.png"
            plt.savefig(sr_path, dpi=150)
            print(f"Saved rolling SR curve to {sr_path}")

            # agreement
            ys2 = [s["agreement"] for s in self.snapshots]
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys2, marker="o", markersize=3, linewidth=1)
            plt.xlabel("Round")
            plt.ylabel("Convention Agreement")
            plt.title("Population Convention Strength")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            agr_path = self.plots_dir / f"{self.cfg.log_prefix}_agreement.png"
            plt.savefig(agr_path, dpi=150)
            print(f"Saved agreement curve to {agr_path}")


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run population simulation with DyadAgent.")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents in population.")
    parser.add_argument("--rounds", type=int, default=20000, help="Number of interaction rounds.")
    parser.add_argument("--log-prefix", type=str, default="pop", help="Prefix for outputs.")
    args = parser.parse_args()

    cfg = PopConfig(
        seed=args.seed,
        n_agents=args.agents,
        n_rounds=args.rounds,
        log_prefix=args.log_prefix,
        # test_start_round=args.test_start,
    )

    sim = PopulationSimulator(cfg)
    sim.run()


if __name__ == "__main__":
    main()