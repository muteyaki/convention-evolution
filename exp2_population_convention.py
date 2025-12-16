"""Population experiment comparing paired vs mixed training modes."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from lexicon import build_entries_with_prior, load_lexicon_config
from population import run_population
from task import load_towers_config, program_to_actions

ModeCurves = Dict[str, Dict[str, List[float]]]


def summarize_records(records: List[Dict[str, float]]) -> Dict[str, List[float]]:
    buckets: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for rec in records:
        key = int(rec.get("round", 0))
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


def summarize_program_lengths_by_task(records: List[Dict[str, float]]) -> Dict[str, Dict[str, List[float]]]:
    per_task: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        task_id = rec.get("tower_id")
        if task_id is None:
            continue
        round_idx = int(rec.get("round", 0))
        per_task[str(task_id)][round_idx].append(rec)

    summary: Dict[str, Dict[str, List[float]]] = {}
    for task_id, round_map in per_task.items():
        rounds = sorted(round_map.keys())
        lengths: List[float] = []
        for r in rounds:
            group = round_map[r]
            if not group:
                continue
            lengths.append(
                sum(len(program_to_actions(g["target_program"])) for g in group) / len(group)
            )
        summary[task_id] = {"rounds": [r + 1 for r in rounds], "program_length": lengths}
    return summary


def summarize_js_history(mean_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract JS divergence curves from mean_belief_history.
    Keys follow evolute_convention structure:
      task_to_program, meaning_to_utterance, utterance_to_meaning
    """
    curves: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"rounds": [], "js": []})
    ordered = sorted(mean_history, key=lambda r: r.get("round", 0))
    for idx, rec in enumerate(ordered):
        round_idx = int(rec.get("round", idx))
        evo = rec.get("evolute_convention", {})
        for key in ("task_to_program", "meaning_to_utterance", "utterance_to_meaning"):
            avg_js = evo.get(key, {}).get("avg_js_divergence")
            if avg_js is None:
                continue
            curves[key]["rounds"].append(round_idx)
            curves[key]["js"].append(avg_js)
    return curves


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
        plt.plot(xs, ys, marker="o", markersize=3, label=mode.capitalize())
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_js_grid(
    js_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]],
    output_path: Path,
) -> None:
    titles = [
        ("task_to_program", "Architect task belief"),
        ("meaning_to_utterance", "Architect utterance belief"),
        ("utterance_to_meaning", "Builder meaning belief"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    label_used: Dict[str, bool] = {}
    for ax, (key, title) in zip(axes, titles):
        for mode, curves in js_by_mode.items():
            data = curves.get(key, {})
            xs = data.get("rounds", [])
            ys = data.get("js", [])
            if xs and ys:
                label = mode.capitalize() if not label_used.get(mode) else None
                ax.plot(xs, ys, marker="o", markersize=2, label=label)
                label_used[mode] = True
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Avg JS Divergence")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_program_length_grid(
    curves_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]],
    task_ids: List[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
    flat_axes = axes.flatten()
    for ax in flat_axes[len(task_ids) :]:
        ax.set_visible(False)

    label_used: Dict[str, bool] = {}
    for idx, task_id in enumerate(task_ids):
        ax = flat_axes[idx]
        for mode, tasks in curves_by_mode.items():
            data = tasks.get(task_id, {})
            xs = data.get("rounds", [])
            ys = data.get("program_length", [])
            if xs and ys:
                label = mode.capitalize() if not label_used.get(mode) else None
                ax.plot(xs, ys, marker="o",markersize=3, label=label)
                label_used[mode] = True
        ax.set_title(f"Task {task_id}")
        ax.set_xlabel("Round")
        if idx % 3 == 0:
            ax.set_ylabel("Avg Program Length")
        ax.grid(True, alpha=0.3)

    # collect legend handles/labels from first visible axis to preserve colors
    first_ax = next((ax for ax in flat_axes if ax.get_visible()), None)
    if first_ax:
        handles, labels = first_ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_mode(
    mode: str,
    n_dyads: int,
    rounds_per_dyad: int,
    seed: int,
    entries,
    lexicon_prior,
    meaning_prior,
    towers_cfg,
) -> Tuple[Dict, Dict]:
    # write intermediate population outputs to a temp dir to avoid clutter
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_population(
            n_dyads=n_dyads,
            entries=entries,
            lexicon_prior=lexicon_prior,
            meaning_prior=meaning_prior,
            towers_cfg=towers_cfg,
            n_rounds_per_dyad=rounds_per_dyad,
            n_generations=1,
            seed=seed,
            results_dir=tmpdir,
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

    results_root = Path(args.results_dir) / "exp2" 
    results_root.mkdir(parents=True, exist_ok=True)
    plots_root = Path(args.plots_dir) / "exp2"
    plots_root.mkdir(parents=True, exist_ok=True)

    metrics: ModeCurves = {}
    summary = {}
    mean_beliefs: Dict[str, List[Dict[str, Any]]] = {}
    program_lengths_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    js_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    dyad_beliefs_by_mode: Dict[str, List[Dict[str, Any]]] = {}

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
        )
        records = generation.get("records", [])
        curves = summarize_records(records)
        metrics[mode] = curves
        program_lengths_by_mode[mode] = summarize_program_lengths_by_task(records)
        js_by_mode[mode] = summarize_js_history(generation.get("mean_belief_history", []))
        dyad_beliefs_by_mode[mode] = generation.get("dyad_belief_history", [])
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
    sorted_task_ids = sorted(towers_cfg.keys(), key=str)
    plot_program_length_grid(
        program_lengths_by_mode,
        sorted_task_ids,
        plots_root / "program_length_by_task.png",
    )
    plot_js_grid(js_by_mode, plots_root / "js_divergence_comparison.png")

    mean_belief_path = results_root / "mean_belief_history.json"
    with mean_belief_path.open("w", encoding="utf-8") as f:
        json.dump(mean_beliefs, f, ensure_ascii=False, indent=2)
    for mode in ("paired", "mixed"):
        path = results_root / f"{mode}_beliefs_history.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(dyad_beliefs_by_mode.get(mode, []), f, ensure_ascii=False, indent=2)

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
