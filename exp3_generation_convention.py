"""Experiment 3: multi-generation population evolution with dyad turnover and round-level tracking."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from lexicon import build_entries_with_prior, load_lexicon_config
from population import Dyad, Population, sample_dyads
from task import load_towers_config, program_to_actions


def _smooth(xs: List[int], ys: List[float], window: int = 3) -> Tuple[List[int], List[float]]:
    if window <= 1 or len(ys) < window:
        return xs, ys
    half = window // 2
    smoothed: List[float] = []
    for i in range(len(ys)):
        start = max(0, i - half)
        end = min(len(ys), i + half + 1)
        segment = ys[start:end]
        smoothed.append(sum(segment) / len(segment))
    return xs, smoothed


def replace_dyads(
    current: List[Dyad],
    n_new: int,
    *,
    entries,
    lexicon_prior,
    meaning_prior,
    towers_cfg,
    seed: int,
) -> List[Dyad]:
    """Replace a subset of dyads with freshly sampled ones; reindex dyad_id sequentially."""
    if n_new <= 0 or not current:
        return current
    rng = random.Random(seed)
    replace_indices = set(rng.sample(range(len(current)), k=min(n_new, len(current))))
    kept = [d for i, d in enumerate(current) if i not in replace_indices]
    new_dyads = sample_dyads(
        n_dyads=n_new,
        entries=entries,
        lexicon_prior=lexicon_prior,
        meaning_prior=meaning_prior,
        towers_cfg=towers_cfg,
        seed=seed + 1,
    )
    merged = kept + new_dyads
    merged = merged[: len(current)]
    reindexed = [Dyad(dyad_id=i, speaker=d.speaker, listener=d.listener) for i, d in enumerate(merged)]
    return reindexed


def run_generation_loop(
    *,
    n_generations: int,
    n_dyads: int,
    rounds_per_dyad: int,
    replace_ratio: float,
    seed: int,
    results_dir: Path,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)
    lex_cfg = load_lexicon_config(LEXICON_CONFIG_PATH)
    entries, lexicon_prior, meaning_prior = build_entries_with_prior(lex_cfg, lam=LENGTH_PRIOR_LAMBDA)

    dyads = sample_dyads(
        n_dyads=n_dyads,
        entries=entries,
        lexicon_prior=lexicon_prior,
        meaning_prior=meaning_prior,
        towers_cfg=towers_cfg,
        seed=seed,
    )

    metrics: List[Dict[str, Any]] = []
    round_metrics: List[Dict[str, Any]] = []
    mean_history_all: List[Dict[str, Any]] = []
    dyad_history_all: List[Dict[str, Any]] = []
    length_by_task: Dict[str, List[Tuple[int, int, float]]] = {tid: [] for tid in towers_cfg.keys()}

    for gen_idx in range(n_generations):
        pop_seed = seed + gen_idx
        population = Population(
            dyads=dyads,
            towers_cfg=towers_cfg,
            lexicon_entries=entries,
            seed=pop_seed,
        )
        generation_result = population.run_generation(
            rounds_per_dyad=rounds_per_dyad,
            training_mode="mixed",
        )
        records = generation_result.get("records", [])
        avg_loss = sum(r["loss"] for r in records) / len(records) if records else 0.0
        metrics.append(
            {
                "generation": gen_idx,
                "mean_accuracy": generation_result.get("mean_accuracy", 0.0),
                "avg_loss": avg_loss,
                "overall_js": generation_result.get("evolute_convention", {}).get("overall_js_divergence", 0.0),
            }
        )

        # aggregate per-round metrics with global round offset
        round_offset = gen_idx * rounds_per_dyad
        round_buckets: Dict[int, List[Dict[str, Any]]] = {}
        for rec in records:
            r = int(rec.get("round", 0))
            round_buckets.setdefault(r, []).append(rec)
        for r, recs in sorted(round_buckets.items()):
            successes = [rc["success"] for rc in recs]
            losses = [rc["loss"] for rc in recs]
            avg_len = sum(len(program_to_actions(rc["target_program"])) for rc in recs) / len(recs)
            round_metrics.append(
                {
                    "generation": gen_idx,
                    "round": round_offset + r,
                    "round_in_gen": r,
                    "mean_accuracy": sum(successes) / len(successes) if successes else 0.0,
                    "avg_loss": sum(losses) / len(losses) if losses else 0.0,
                    "avg_program_length": avg_len,
                }
            )
            task_groups: Dict[str, List[int]] = {}
            for rc in recs:
                tid = rc.get("tower_id")
                if tid is None:
                    continue
                task_groups.setdefault(str(tid), []).append(len(program_to_actions(rc["target_program"])))
            for tid, vals in task_groups.items():
                length_by_task.setdefault(tid, []).append((gen_idx, r, sum(vals) / len(vals)))

        # mean belief and dyad belief history with global round offset
        for entry in generation_result.get("mean_belief_history", []):
            entry_copy = copy.deepcopy(entry)
            entry_copy["round"] = round_offset + int(entry.get("round", 0))
            entry_copy["generation"] = gen_idx
            entry_copy["round_in_gen"] = int(entry.get("round", 0))
            mean_history_all.append(entry_copy)
        for entry in generation_result.get("dyad_belief_history", []):
            entry_copy = copy.deepcopy(entry)
            entry_copy["round"] = round_offset + int(entry.get("round", 0))
            entry_copy["generation"] = gen_idx
            entry_copy["round_in_gen"] = int(entry.get("round", 0))
            dyad_history_all.append(entry_copy)

        # Prepare next generation population with turnover
        if gen_idx < n_generations - 1:
            n_replace = max(1, int(n_dyads * replace_ratio))
            dyads = replace_dyads(
                dyads,
                n_replace,
                entries=entries,
                lexicon_prior=lexicon_prior,
                meaning_prior=meaning_prior,
                towers_cfg=towers_cfg,
                seed=rng.randint(0, 1_000_000_000),
            )

    results_obj = {
        "metrics": metrics,
        "round_metrics": round_metrics,
        "mean_belief_history": mean_history_all,
        "dyad_belief_history": dyad_history_all,
        "length_by_task": length_by_task,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "exp3_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f, ensure_ascii=False, indent=2)
    return results_obj


def _plot_metric_by_generation(
    round_metrics: List[Dict[str, Any]], key: str, ylabel: str, title: str, rounds_per_gen: int, out_path: Path
) -> None:
    plt.figure(figsize=(7, 4))
    by_gen: Dict[int, List[Tuple[int, float]]] = {}
    for rm in round_metrics:
        g = int(rm.get("generation", 0))
        r = int(rm.get("round_in_gen", rm.get("round", 0)))
        by_gen.setdefault(g, []).append((r, rm.get(key, 0.0)))
    for g, vals in sorted(by_gen.items()):
        vals_sorted = sorted(vals, key=lambda x: x[0])
        xs = [v[0] + 1 for v in vals_sorted]
        ys = [v[1] for v in vals_sorted]
        xs, ys = _smooth(xs, ys)
        plt.plot(xs, ys, label=f"Gen {g + 1}")
    plt.xlim(1, rounds_per_gen)
    plt.xlabel("Round in Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_length_by_task(length_by_task: Dict[str, List[Tuple[int, int, float]]], rounds_per_gen: int, out_path: Path) -> None:
    task_ids = sorted(length_by_task.keys(), key=str)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for idx, tid in enumerate(task_ids):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        vals = length_by_task.get(tid, [])
        by_gen: Dict[int, List[Tuple[int, float]]] = {}
        for g, r, val in vals:
            by_gen.setdefault(g, []).append((r, val))
        for g, arr in sorted(by_gen.items()):
            arr_sorted = sorted(arr, key=lambda x: x[0])
            xs = [v[0] + 1 for v in arr_sorted]
            ys = [v[1] for v in arr_sorted]
            xs, ys = _smooth(xs, ys)
            ax.plot(xs, ys, label=f"Gen {g + 1}")
        ax.set_xlim(1, rounds_per_gen)
        ax.set_title(f"Task {tid}")
        ax.grid(True, alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel("Avg Program Length")
        ax.set_xlabel("Round in Gen")
    for ax in axes_flat[len(task_ids) :]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_js_over_rounds(mean_history: List[Dict[str, Any]], rounds_per_gen: int, out_path: Path) -> None:
    keys = [
        ("task_to_program", "Task → Program"),
        ("meaning_to_utterance", "Meaning → Utterance"),
        ("utterance_to_meaning", "Utterance → Meaning"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, (k, label) in zip(axes, keys):
        by_gen: Dict[int, List[Tuple[int, float]]] = {}
        for entry in mean_history:
            g = int(entry.get("generation", 0))
            r = int(entry.get("round_in_gen", entry.get("round", 0)))
            val = entry.get("evolute_convention", {}).get(k, {}).get("avg_js_divergence")
            if val is None:
                continue
            by_gen.setdefault(g, []).append((r, val))
        for g, vals in sorted(by_gen.items()):
            vals_sorted = sorted(vals, key=lambda x: x[0])
            xs = [v[0] + 1 for v in vals_sorted]
            ys = [v[1] for v in vals_sorted]
            xs, ys = _smooth(xs, ys)
            ax.plot(xs, ys, label=f"Gen {g + 1}")
        ax.set_xlim(1, rounds_per_gen)
        ax.set_title(label)
        ax.set_xlabel("Round in Gen")
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Avg JS Divergence")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-generation population evolution with dyad turnover.")
    parser.add_argument("--generations", type=int, default=GEN_POP, help="Number of generations to run.")
    parser.add_argument("--dyads", type=int, default=N_DYADS, help="Number of dyads per generation.")
    parser.add_argument(
        "--rounds-per-dyad",
        type=int,
        default=ROUNDS_PER_DYAD,
        help="Rounds per dyad per generation.",
    )
    parser.add_argument(
        "--replace-ratio",
        type=float,
        default=NEWPOP_RATIO,
        help="Fraction of dyads to replace between generations (0–1).",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "exp3", help="Directory for outputs.")
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR / "exp3", help="Directory for plots.")
    args = parser.parse_args()

    replace_ratio = min(max(args.replace_ratio, 0.0), 1.0)

    results = run_generation_loop(
        n_generations=args.generations,
        n_dyads=args.dyads,
        rounds_per_dyad=args.rounds_per_dyad,
        replace_ratio=replace_ratio,
        seed=GLOBAL_SEED,
        results_dir=args.results_dir,
    )

    # save round-level histories
    mean_path = args.results_dir / "mean_belief_history.json"
    dyad_path = args.results_dir / "dyad_belief_history.json"
    with mean_path.open("w", encoding="utf-8") as f:
        json.dump(results["mean_belief_history"], f, ensure_ascii=False, indent=2)
    with dyad_path.open("w", encoding="utf-8") as f:
        json.dump(results["dyad_belief_history"], f, ensure_ascii=False, indent=2)

    # plots over rounds
    plots_dir = args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    round_metrics = sorted(results["round_metrics"], key=lambda r: r["round"])
    _plot_metric_by_generation(
        round_metrics, "mean_accuracy", "Mean Accuracy", "Accuracy over Rounds", args.rounds_per_dyad, plots_dir / "accuracy.png"
    )
    _plot_metric_by_generation(
        round_metrics, "avg_loss", "Average Loss", "Loss over Rounds", args.rounds_per_dyad, plots_dir / "loss.png"
    )
    _plot_length_by_task(results["length_by_task"], args.rounds_per_dyad, plots_dir / "program_length_by_task.png")
    _plot_js_over_rounds(results["mean_belief_history"], args.rounds_per_dyad, plots_dir / "js_divergence_over_rounds.png")

    print("=== exp3 generation convention ===")
    for m in results["metrics"]:
        print(
            f"Gen {m['generation']+1}: mean_acc={m['mean_accuracy']:.3f}, "
            f"avg_loss={m['avg_loss']:.3f}, overall_js={m['overall_js']:.3f}"
        )
    print(f"Saved summary to {args.results_dir / 'exp3_summary.json'}")
    print(f"Saved mean belief history to {mean_path}")
    print(f"Saved dyad belief history to {dyad_path}")
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
