"""Visualize exp2 mean belief history as heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

from config import TOWERS_CONFIG_PATH 
from task import load_towers_config, program_length  


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    Z = sum(dist.values())
    if Z <= 0:
        return {k: 0.0 for k in dist}
    return {k: v / Z for k, v in dist.items()}


def _shorten_from_meaning(m: str) -> str:
    mapping = {
        "horizontal": "h",
        "vertical": "v",
        "left": "l",
        "right": "r",
        "chunk_8": "8-shaped",
        "chunk_C": "C-shaped",
        "chunk_L": "L-shaped",
        "chunk_Pi": "Pi-shaped",
    }
    out = m
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def load_mean_history(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_round_index(entries: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {e.get("round", i): e for i, e in enumerate(entries)}


def _collect_meaning_utterance_sets(mean_history: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    meanings = set()
    utterances = set()
    for mode_entries in mean_history.values():
        for entry in mode_entries:
            m2u = entry.get("evolute_convention", {}).get("meaning_to_utterance", {}).get("mean", {})
            meanings.update(m2u.keys())
            for dist in m2u.values():
                utterances.update(dist.keys())
            u2m = entry.get("evolute_convention", {}).get("utterance_to_meaning", {}).get("mean", {})
            utterances.update(u2m.keys())
            for dist in u2m.values():
                meanings.update(dist.keys())
    meaning_list = sorted(meanings, key=str)
    meaning_order = {m: idx for idx, m in enumerate(meaning_list)}
    return {"meanings": meaning_list, "meaning_order": meaning_order, "utterances": sorted(utterances)}


def _infer_utterance_meaning(mean_history: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    """Infer utterance->meaning using highest prob across all mode/round meaning_to_utterance means."""
    best: Dict[str, Any] = {}
    for mode_entries in mean_history.values():
        for entry in mode_entries:
            m2u = entry.get("evolute_convention", {}).get("meaning_to_utterance", {}).get("mean", {})
            for m, dist in m2u.items():
                for u, p in dist.items():
                    if u not in best or p > best[u][1]:
                        best[u] = (m, p)
    return {u: mp[0] for u, mp in best.items()}


def _get_round_entry(round_idx: int, round_index: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    return round_index.get(round_idx, {})


def plot_m2u_heatmap(mean_history: Dict[str, List[Dict[str, Any]]], out_path: Path) -> None:
    rounds_to_plot = [0, 20, 40, 60]
    sets = _collect_meaning_utterance_sets(mean_history)
    meanings: List[str] = sets["meanings"]
    meaning_order: Dict[str, int] = sets["meaning_order"]
    utterances: List[str] = sets["utterances"]
    u_to_m = _infer_utterance_meaning(mean_history)
    utterances_sorted = sorted(
        utterances, key=lambda u: (meaning_order.get(u_to_m.get(u, ""), len(meanings)), u)
    )
    if not meanings or not utterances_sorted:
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    vmax = 1.0
    vmin = 0.0
    modes = ["paired", "mixed"]
    im = None
    for row, mode in enumerate(modes):
        round_index = _build_round_index(mean_history.get(mode, []))
        for col, r in enumerate(rounds_to_plot):
            ax = axes[row][col]
            entry = _get_round_entry(r, round_index)
            dist_m2u = entry.get("evolute_convention", {}).get("meaning_to_utterance", {}).get("mean", {})
            matrix = np.array(
                [[_normalize(dist_m2u.get(m, {})).get(u, 0.0) for u in utterances_sorted] for m in meanings]
            )
            im = ax.imshow(matrix, aspect="equal", cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(f"{mode.capitalize()} round {r}")
            if row == len(modes) - 1:
                ax.set_xticks(range(len(utterances_sorted)))
                ax.set_xticklabels(
                    [_shorten_from_meaning(u_to_m.get(u, u)) for u in utterances_sorted],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
            else:
                ax.set_xticks([])
            ax.set_yticks(range(len(meanings)))
            ax.set_yticklabels(meanings, fontsize=8)
            ax.set_xlabel("Utterance")
            ax.set_ylabel("Meaning")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.3, location="bottom")
        cbar.set_label("Probability")
    fig.suptitle("P(utterance | meaning) heatmaps (paired vs mixed)", fontsize=14)
    fig.tight_layout(rect=[0, 0.25, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_u2m_heatmap(mean_history: Dict[str, List[Dict[str, Any]]], out_path: Path) -> None:
    rounds_to_plot = [0, 20, 40, 60]
    sets = _collect_meaning_utterance_sets(mean_history)
    meanings: List[str] = sets["meanings"]
    meaning_order: Dict[str, int] = sets["meaning_order"]
    utterances: List[str] = sets["utterances"]
    u_to_m = _infer_utterance_meaning(mean_history)
    utterances_sorted = sorted(
        utterances, key=lambda u: (meaning_order.get(u_to_m.get(u, ""), len(meanings)), u)
    )
    if not meanings or not utterances_sorted:
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    vmax = 1.0
    vmin = 0.0
    modes = ["paired", "mixed"]
    im = None
    for row, mode in enumerate(modes):
        round_index = _build_round_index(mean_history.get(mode, []))
        for col, r in enumerate(rounds_to_plot):
            ax = axes[row][col]
            entry = _get_round_entry(r, round_index)
            u2m = entry.get("evolute_convention", {}).get("utterance_to_meaning", {}).get("mean", {})
            matrix_rows: List[List[float]] = []
            for m in meanings:
                row_vals: List[float] = []
                for u in utterances_sorted:
                    dist = _normalize(u2m.get(u, {}))
                    row_vals.append(dist.get(m, 0.0))
                matrix_rows.append(row_vals)
            matrix = np.array(matrix_rows)
            im = ax.imshow(matrix, aspect="equal", cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(f"{mode.capitalize()} round {r}")
            if row == len(modes) - 1:
                ax.set_xticks(range(len(utterances_sorted)))
                ax.set_xticklabels(
                    [_shorten_from_meaning(u_to_m.get(u, u)) for u in utterances_sorted],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
            else:
                ax.set_xticks([])
            ax.set_yticks(range(len(meanings)))
            ax.set_yticklabels(meanings, fontsize=8)
            ax.set_xlabel("Utterance")
            ax.set_ylabel("Meaning")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.3, location="bottom")
        cbar.set_label("Probability")
    fig.suptitle("P(meaning | utterance) heatmaps (paired vs mixed)", fontsize=14)
    fig.tight_layout(rect=[0, 0.25, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_task_heatmap(mean_history: Dict[str, List[Dict[str, Any]]], towers_cfg: Dict[str, Any], out_path: Path) -> None:
    modes = ["paired", "mixed"]
    task_ids = sorted(towers_cfg.keys(), key=str)
    rounds = list(range(0, 61))

    fig, axes = plt.subplots(2, 6, figsize=(18, 8), sharey=False)
    im = None
    for row, mode in enumerate(modes):
        round_index = _build_round_index(mean_history.get(mode, []))
        for col, task_id in enumerate(task_ids):
            if col >= 6:
                break
            ax = axes[row][col]
            programs = sorted(towers_cfg.get(task_id, {}).get("correct_programs", []), key=program_length)
            if not programs:
                ax.axis("off")
                continue
            matrix_rows: List[List[float]] = []
            for p in programs:
                row_vals: List[float] = []
                for r in rounds:
                    entry = _get_round_entry(r, round_index)
                    dist = entry.get("evolute_convention", {}).get("task_to_program", {}).get("mean", {}).get(task_id, {})
                    row_vals.append(_normalize(dist).get(p, 0.0))
                matrix_rows.append(row_vals)
            matrix = np.array(matrix_rows)
            im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0, origin="lower")
            ax.set_title(f"{mode.capitalize()} task {task_id}")
            ax.set_yticks(range(len(programs)))
            ax.set_yticklabels([f"L={program_length(p)}" for p in programs], fontsize=7)
            step = max(1, len(rounds) // 6)
            xticks = list(range(0, len(rounds), step))
            ax.set_xticks(xticks)
            ax.set_xticklabels([rounds[i] for i in xticks], rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Round")
            if col == 0:
                ax.set_ylabel("Program")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.25, location="bottom")
        cbar.set_label("Probability")
    fig.suptitle("P(program | task) over rounds (paired vs mixed)", fontsize=14)
    fig.tight_layout(rect=[0, 0.2, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize exp2 mean_belief_history.json as heatmaps.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "exp2" / "mean_belief_history.json",
        help="Path to mean_belief_history.json",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("plots") / "exp2", help="Directory to save figures")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mean_history = load_mean_history(args.input)
    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)

    plot_m2u_heatmap(mean_history, args.out_dir / "architect_m2u_heatmap.png")
    plot_u2m_heatmap(mean_history, args.out_dir / "builder_u2m_heatmap.png")
    plot_task_heatmap(mean_history, towers_cfg, args.out_dir / "architect_task_heatmap.png")
    print(f"Saved heatmaps to {args.out_dir}")


if __name__ == "__main__":
    main()
