"""Population experiment comparing paired vs mixed training modes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from lexicon import build_entries_with_prior, load_lexicon_config
from population_core import run_population
from task import load_towers_config, program_to_actions

ModeCurves = Dict[str, Dict[str, List[float]]]


def _compute_round_key(mode: str, record_round: int, n_dyads: int) -> int:
    if mode == "mixed":
        return record_round // max(n_dyads, 1)
    return record_round


def summarize_records(records: List[Dict[str, float]], mode: str, n_dyads: int) -> Dict[str, List[float]]:
    buckets: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for rec in records:
        key = _compute_round_key(mode, int(rec.get("round", 0)), n_dyads)
        buckets[key].append(rec)

    if not buckets:
        return {"rounds": [], "loss": [], "accuracy": [], "program_length": []}

    rounds = sorted(buckets.keys())
    losses: List[float] = []
    accuracies: List[float] = []
    lengths: List[float] = []
    for r in rounds:
        group = buckets[r]
        if not group:
            continue
        losses.append(sum(g["loss"] for g in group) / len(group))
        accuracies.append(sum(g["success"] for g in group) / len(group))
        lengths.append(
            sum(len(program_to_actions(g["target_program"])) for g in group) / len(group)
        )

    return {
        "rounds": [r + 1 for r in rounds],
        "loss": losses,
        "accuracy": accuracies,
        "program_length": lengths,
    }


def plot_curve(
    curves: ModeCurves,
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 4))
    for mode, values in curves.items():
        xs = values.get("rounds", [])
        ys = values.get(metric_key, [])
        if not xs or not ys:
            continue
        plt.plot(xs, ys, marker="o", label=mode.capitalize())
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_mode(
    mode: str,
    n_dyads: int,
    rounds_per_dyad: int,
    seed: int,
    entries,
    lexicon_prior,
    meaning_prior,
    towers_cfg,
    base_results_dir: Path,
) -> Tuple[Dict, Dict]:
    mode_results_dir = base_results_dir / f"exp2_{mode}"
    mode_results_dir.mkdir(parents=True, exist_ok=True)
    results = run_population(
        n_dyads=n_dyads,
        entries=entries,
        lexicon_prior=lexicon_prior,
        meaning_prior=meaning_prior,
        towers_cfg=towers_cfg,
        n_rounds_per_dyad=rounds_per_dyad,
        n_generations=1,
        seed=seed,
        results_dir=str(mode_results_dir),
        training_mode=mode,
    )
    generation = results["generations"][-1] if results["generations"] else {}
    return results, generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare population training modes.")
    parser.add_argument("--dyads", type=int, default=N_DYADS, help="Number of dyads in the population.")
    parser.add_argument(
        "--rounds-per-dyad",
        type=int,
        default=ROUNDS_PER_DYAD,
        help="Rounds per dyad in a generation.",
    )
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed.")
    parser.add_argument("--results-dir", type=str, default="results", help="Base directory for results.")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory to store comparison plots.")
    args = parser.parse_args()

    lex_cfg = load_lexicon_config(LEXICON_CONFIG_PATH)
    entries, lexicon_prior, meaning_prior = build_entries_with_prior(lex_cfg, lam=LENGTH_PRIOR_LAMBDA)
    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    plots_root = Path(args.plots_dir) / "exp2"
    plots_root.mkdir(parents=True, exist_ok=True)

    metrics: ModeCurves = {}
    summary = {}
    mean_beliefs: Dict[str, List[Dict[str, Any]]] = {}

    for mode in ("paired", "mixed"):
        results, generation = run_mode(
            mode=mode,
            n_dyads=args.dyads,
            rounds_per_dyad=args.rounds_per_dyad,
            seed=args.seed,
            entries=entries,
            lexicon_prior=lexicon_prior,
            meaning_prior=meaning_prior,
            towers_cfg=towers_cfg,
            base_results_dir=results_root,
        )
        records = generation.get("records", [])
        curves = summarize_records(records, mode, args.dyads)
        metrics[mode] = curves
        avg_loss = sum(r["loss"] for r in records) / len(records) if records else 0.0
        avg_acc = sum(r["success"] for r in records) / len(records) if records else 0.0
        summary[mode] = {
            "mean_accuracy": generation.get("mean_accuracy", 0.0),
            "avg_loss": avg_loss,
            "overall_js": generation.get("evolute_convention", {}).get("overall_js_divergence", 0.0),
        }
        mean_beliefs[mode] = generation.get("mean_belief_history", [])

    plot_curve(metrics, "loss", "Average Loss", "Population Loss Comparison", plots_root / "loss_comparison.png")
    plot_curve(
        metrics,
        "accuracy",
        "Average Accuracy",
        "Population Accuracy Comparison",
        plots_root / "accuracy_comparison.png",
    )
    plot_curve(
        metrics,
        "program_length",
        "Average Program Length",
        "Population Program Length Comparison",
        plots_root / "program_length_comparison.png",
    )

    mean_belief_path = results_root / "mean_belief_history.json"
    with mean_belief_path.open("w", encoding="utf-8") as f:
        json.dump(mean_beliefs, f, ensure_ascii=False, indent=2)

    print("=== exp2 population comparison ===")
    for mode, stats in summary.items():
        print(
            f"Mode={mode}: mean_accuracy={stats['mean_accuracy']:.3f}, "
            f"avg_loss={stats['avg_loss']:.3f}, overall_js={stats['overall_js']:.3f}"
        )
    print(f"Saved comparison plots to {plots_root}")
    print(f"Saved mean belief summary to {mean_belief_path}")


if __name__ == "__main__":
    main()
