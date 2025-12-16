"""Dyad demo: initialize a single Architect–Builder pair and track learning over rounds."""

import argparse
import random
from typing import Dict, List
import math
import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import json
import copy


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    Z = sum(dist.values())
    if Z <= 0:
        return {k: 0.0 for k in dist}
    return {k: v / Z for k, v in dist.items()}

from agents import AgentConfig, DyadAgent
from config import *
from task import load_towers_config, program_length, program_to_actions
from lexicon import load_lexicon_config, build_entries_with_prior


def run_dyad_demo(
    n_rounds: int,
    length_prior_lambda: float,
    seed: int,
    log_prefix: str = "sampling_log",
) -> None:
    random.seed(seed)

    plots_dir = Path("plots") / "exp1"
    plots_dir.mkdir(exist_ok=True)
    results_dir = Path("results") / "exp1"
    results_dir.mkdir(exist_ok=True)

    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)
    lex_cfg = load_lexicon_config(LEXICON_CONFIG_PATH)
    lexicon_entries, lexicon_prior, meaning_prior = build_entries_with_prior(
        lex_cfg, lam=length_prior_lambda
    )

    architect_log = results_dir / f"{log_prefix}_architect.json" if log_prefix else None
    builder_log = results_dir / f"{log_prefix}_builder.json" if log_prefix else None

    cfg_A = AgentConfig(seed=seed)
    cfg_B = AgentConfig(seed=seed + 1)
    architect = DyadAgent(
        lexicon_entries=lexicon_entries,
        lexicon_prior=lexicon_prior,
        meaning_prior=meaning_prior,
        towers_cfg=towers_cfg,
        cfg=cfg_A,
        log_path=architect_log,
    )
    builder = DyadAgent(
        lexicon_entries=lexicon_entries,
        lexicon_prior=lexicon_prior,
        meaning_prior=meaning_prior,
        towers_cfg=towers_cfg,
        cfg=cfg_B,
        log_path=builder_log,
    )

    losses = []
    successes = []
    tower_ids = list(towers_cfg.keys())
    if not tower_ids:
        raise ValueError("No towers available in configuration.")
    program_lengths_by_task = {tid: [] for tid in tower_ids}
    belief_history = []

    # Pre-generate a tower sampling schedule so each ID appears roughly equally often.
    cycles = math.ceil(n_rounds / len(tower_ids))
    tower_schedule: List[str] = []
    for _ in range(cycles):
        shuffled = tower_ids[:]
        random.shuffle(shuffled)
        tower_schedule.extend(shuffled)
    tower_schedule = tower_schedule[:n_rounds]

    def snapshot(agent: DyadAgent) -> Dict[str, Dict[str, Dict[str, float]]]:
        meaning_to_utterance = {m: _normalize(agent.belief_m2u.get(m, {})) for m in agent.meanings}
        task_to_program = {task: _normalize(probs) for task, probs in agent.task_program_belief.items()}
        return {
            "meaning_to_utterance": meaning_to_utterance,
            "task_to_program": task_to_program,
        }

    belief_history.append(
        {
            "round": -1,
            "architect": snapshot(architect),
            "builder": snapshot(builder),
        }
    )

    for t in range(n_rounds):
        tower_id = tower_schedule[t]
        # tower_id = "PiC"
        sampled_program, utterance_seq = architect.produce_message_for_task(tower_id)
        target_tokens = program_to_actions(sampled_program)
        program_len = len(target_tokens)
        program_lengths_by_task[tower_id].append(program_len)
        decoded_tokens = builder.interpret_message(utterance_seq)

        guess_program = " ".join(decoded_tokens) if decoded_tokens else None
        success = 1 if guess_program == sampled_program else 0
        min_len = min(len(target_tokens), len(decoded_tokens))
        mismatches = sum(1 for a, b in zip(target_tokens[:min_len], decoded_tokens[:min_len]) if a != b)
        length_diff = abs(len(target_tokens) - len(decoded_tokens))
        raw_loss = mismatches + length_diff
        total_len = program_len if program_len > 0 else 1
        loss = raw_loss / total_len
        losses.append(loss)
        successes.append(success)

        print(
            f"[round {t:02d}] tower={tower_id} target={sampled_program} | "
            f"sampled_program={sampled_program} | utterance={utterance_seq} | guess={guess_program} | "
            f"len={program_len} | loss={loss:.3f} | success={success}"
        )

        architect.observe_interaction(tower_id, sampled_program, utterance_seq, decoded_tokens)
        builder.observe_interaction(tower_id, sampled_program, utterance_seq, decoded_tokens)

        # save belief log
        belief_history.append(
            {
                "round": t,
                "architect": snapshot(architect),
                "builder": snapshot(builder),
            }
        )

    avg_loss = sum(losses) / len(losses)
    acc = sum(successes) / len(successes)
    print("=== exp1 dyad convention ===")
    print(f"[dyad demo] rounds={n_rounds}, avg loss={avg_loss:.3f}, acc={acc:.3f}")

    # plot loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(n_rounds), losses, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Loss (mismatch / length)")
    plt.title("Dyad Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = plots_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=150)
    print(f"Saved loss curve to {loss_path}")

    # plot accuracy curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(n_rounds), successes, marker="o", color="green")
    plt.xlabel("Round")
    plt.ylabel("Per-round Accuracy")
    plt.title("Dyad Per-round Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = plots_dir / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=150)
    print(f"Saved accuracy curve to {acc_path}")

    # plot program length curve by task
    plt.figure(figsize=(6, 4))
    for task_id in sorted(program_lengths_by_task.keys()):
        lengths = program_lengths_by_task[task_id]
        if not lengths:
            continue
        xs = list(range(1, len(lengths) + 1))
        plt.plot(xs, lengths, marker="o", label=str(task_id))
    plt.xlabel("Occurrence index for task")
    plt.ylabel("Program Length (actions)")
    plt.title("Program Length by Task")
    plt.legend(title="Task ID")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    length_path = plots_dir / "program_length_curve.png"
    plt.savefig(length_path, dpi=150)
    print(f"Saved program length curve to {length_path}")

    # save distribution of every trail
    belief_path = results_dir / "belief_history.json"
    with open(belief_path, "w", encoding="utf-8") as f:
        json.dump(belief_history, f, ensure_ascii=False, indent=2)
    print(f"Saved belief history to {belief_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single dyad for a few rounds and report learning metrics.")
    parser.add_argument("--rounds", type=int, default=50, help="Number of dyad rounds.")
    parser.add_argument("--lambda-length", type=float, default=LENGTH_PRIOR_LAMBDA, help="Length prior weight λ.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed.")
    parser.add_argument("--log-prefix", type=str, default="sampling_log", help="Prefix for agent sampling logs (JSON).")
    args = parser.parse_args()

    run_dyad_demo(
        n_rounds=args.rounds,
        length_prior_lambda=args.lambda_length,
        seed=args.seed,
        log_prefix=args.log_prefix,
    )


if __name__ == "__main__":
    main()
