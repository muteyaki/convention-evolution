"""Visualize Exp3 mean belief history as heatmaps (generation comparison)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import TOWERS_CONFIG_PATH  # noqa: E402
from task import load_towers_config, program_length  # noqa: E402


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


def load_mean_history(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _group_by_generation(
    entries: List[Dict[str, Any]], generations: List[int]
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for g in generations:
        grouped[str(g)] = [e for e in entries if int(e.get("generation", -1)) == g]
    return grouped


def _build_round_index(entries: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(e.get("round_in_gen", e.get("round", i))): e for i, e in enumerate(entries)}


def _aligned_max_round(
    mean_history: Dict[str, List[Dict[str, Any]]],
    gens: List[str],
    *,
    allow_missing: bool = False,
) -> int:
    max_rounds: List[int] = []
    for gen in gens:
        rounds = [
            int(e.get("round_in_gen", e.get("round")))
            for e in mean_history.get(gen, [])
            if e.get("round_in_gen", e.get("round")) is not None
        ]
        if not rounds:
            if allow_missing:
                continue
            return -1
        max_rounds.append(max(rounds))
    if not max_rounds:
        return -1
    return min(max_rounds)


def _round_slices(max_round: int, n_slices: int) -> List[int]:
    if n_slices <= 0:
        return []
    if max_round <= 0:
        return [0 for _ in range(n_slices)]
    values = np.linspace(0, max_round, num=n_slices)
    rounds = [int(round(v)) for v in values]
    rounds[0] = 0
    rounds[-1] = max_round
    for i in range(1, len(rounds)):
        rounds[i] = max(rounds[i], rounds[i - 1])
    return rounds


def _gen_label(gen: str) -> str:
    return f"Gen {int(gen) + 1}"


def _collect_meaning_utterance_sets(mean_history: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    meanings = set()
    utterances = set()
    for entries in mean_history.values():
        for entry in entries:
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
    """Infer utterance->meaning using max prob across all entries."""
    best: Dict[str, Any] = {}
    for entries in mean_history.values():
        for entry in entries:
            m2u = entry.get("evolute_convention", {}).get("meaning_to_utterance", {}).get("mean", {})
            for m, dist in m2u.items():
                for u, p in dist.items():
                    if u not in best or p > best[u][1]:
                        best[u] = (m, p)
    return {u: mp[0] for u, mp in best.items()}


def _get_round_entry(round_idx: int, round_index: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    return round_index.get(round_idx, {})


def plot_m2u_heatmap(mean_history: Dict[str, List[Dict[str, Any]]], out_path: Path) -> None:
    gens = [str(g) for g in range(5)]
    max_round = _aligned_max_round(mean_history, gens, allow_missing=True)
    if max_round < 0:
        return
    rounds_to_plot = _round_slices(max_round, 5)
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

    fig, axes = plt.subplots(
        len(gens),
        len(rounds_to_plot),
        figsize=(4 * len(rounds_to_plot), 3.0 * len(gens)),
        sharex=True,
        sharey=True,
    )
    vmax = 1.0
    vmin = 0.0
    im = None
    for row, gen in enumerate(gens):
        gen_entries = mean_history.get(gen, [])
        if not gen_entries:
            for col in range(len(rounds_to_plot)):
                ax = axes[row][col]
                ax.axis("off")
                if col == 0:
                    ax.set_title(f"{_gen_label(gen)} (no data)")
            continue
        round_index = _build_round_index(gen_entries)
        for col, r in enumerate(rounds_to_plot):
            ax = axes[row][col]
            entry = _get_round_entry(r, round_index)
            dist_m2u = entry.get("evolute_convention", {}).get("meaning_to_utterance", {}).get("mean", {})
            matrix = np.array(
                [[_normalize(dist_m2u.get(m, {})).get(u, 0.0) for u in utterances_sorted] for m in meanings]
            )
            im = ax.imshow(matrix, aspect="equal", cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(f"{_gen_label(gen)} round {r}")
            if row == len(gens) - 1:
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
            if row == len(gens) - 1:
                ax.set_xlabel("Utterance")
            if col == 0:
                ax.set_ylabel("Meaning")
    fig.suptitle("P(utterance | meaning) heatmaps (Gen 1–5)", fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.22, wspace=0.05, hspace=0.25)
    if im is not None:
        cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Probability")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_u2m_heatmap(mean_history: Dict[str, List[Dict[str, Any]]], out_path: Path) -> None:
    gens = [str(g) for g in range(5)]
    max_round = _aligned_max_round(mean_history, gens, allow_missing=True)
    if max_round < 0:
        return
    rounds_to_plot = _round_slices(max_round, 5)
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

    fig, axes = plt.subplots(
        len(gens),
        len(rounds_to_plot),
        figsize=(4 * len(rounds_to_plot), 3.0 * len(gens)),
        sharex=True,
        sharey=True,
    )
    vmax = 1.0
    vmin = 0.0
    im = None
    for row, gen in enumerate(gens):
        gen_entries = mean_history.get(gen, [])
        if not gen_entries:
            for col in range(len(rounds_to_plot)):
                ax = axes[row][col]
                ax.axis("off")
                if col == 0:
                    ax.set_title(f"{_gen_label(gen)} (no data)")
            continue
        round_index = _build_round_index(gen_entries)
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
            ax.set_title(f"{_gen_label(gen)} round {r}")
            if row == len(gens) - 1:
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
            if row == len(gens) - 1:
                ax.set_xlabel("Utterance")
            if col == 0:
                ax.set_ylabel("Meaning")
    fig.suptitle("P(meaning | utterance) heatmaps (Gen 1–5)", fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.22, wspace=0.05, hspace=0.25)
    if im is not None:
        cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Probability")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_task_heatmap(mean_history: Dict[str, List[Dict[str, Any]]], towers_cfg: Dict[str, Any], out_path: Path) -> None:
    gens = [str(g) for g in range(5)]
    task_ids = sorted(towers_cfg.keys(), key=str)
    max_round = _aligned_max_round(mean_history, gens, allow_missing=True)
    if max_round < 0:
        return
    rounds = list(range(max_round + 1))

    fig, axes = plt.subplots(len(gens), 6, figsize=(18, 3.0 * len(gens)), sharey=False)
    im = None
    for row, gen in enumerate(gens):
        gen_entries = mean_history.get(gen, [])
        if not gen_entries:
            for col in range(min(6, len(task_ids))):
                ax = axes[row][col]
                ax.axis("off")
                if col == 0:
                    ax.set_title(f"{_gen_label(gen)} (no data)")
            continue

        round_index = _build_round_index(gen_entries)
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
            ax.set_title(f"{_gen_label(gen)} task {task_id}")
            ax.set_yticks(range(len(programs)))
            ax.set_yticklabels([f"L={program_length(p)}" for p in programs], fontsize=7)
            step = max(1, len(rounds) // 6)
            xticks = list(range(0, len(rounds), step))
            ax.set_xticks(xticks)
            ax.set_xticklabels([rounds[i] for i in xticks], rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Round")
            if col == 0:
                ax.set_ylabel(f"{_gen_label(gen)}\nProgram")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.2, location="bottom")
        cbar.set_label("Probability")
    fig.suptitle("P(program | task) over rounds (Gen 1–5)", fontsize=14)
    fig.tight_layout(rect=[0, 0.18, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Exp3 mean_belief_history.json as heatmaps (Gen 1–5)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "exp3" / "mean_belief_history.json",
        help="Path to mean_belief_history.json",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("plots") / "exp3", help="Directory to save figures")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_entries: List[Dict[str, Any]] = load_mean_history(args.input)
    mean_history = _group_by_generation(all_entries, generations=list(range(5)))
    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)

    plot_m2u_heatmap(mean_history, args.out_dir / "architect_m2u_heatmap.png")
    plot_u2m_heatmap(mean_history, args.out_dir / "builder_u2m_heatmap.png")
    plot_task_heatmap(mean_history, towers_cfg, args.out_dir / "architect_task_heatmap.png")
    print(f"Saved heatmaps to {args.out_dir}")


if __name__ == "__main__":
    main()
