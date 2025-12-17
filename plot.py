"""Visualize dyad belief history as heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def _slice_rounds(history: List[Dict], step: int, max_round: int, max_slices: int) -> List[int]:
    rounds = sorted({entry["round"] for entry in history if entry["round"] >= 0})
    targets = [r for r in range(0, max_round + 1, step)]
    selected = [r for r in targets if r in rounds][:max_slices]
    return selected


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


def _infer_utterance_meaning(history: List[Dict], agent_key: str) -> Dict[str, str]:
    """Infer utterance->meaning by taking the meaning with highest prob for that utterance across rounds."""
    best: Dict[str, Tuple[str, float]] = {}
    for entry in history:
        m2u = entry[agent_key]["meaning_to_utterance"]
        for m, dist in m2u.items():
            for u, p in dist.items():
                if u not in best or p > best[u][1]:
                    best[u] = (m, p)
    return {u: mp[0] for u, mp in best.items()}


def load_history(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"belief history not found at {path}. Run exp1_dyad_convention.py first or pass --input <path>."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_round_index(history: List[Dict]) -> Dict[int, Dict]:
    return {entry["round"]: entry for entry in history if entry["round"] >= 0}


def _meaning_utterance_slice_grids(
    history: List[Dict],
    agent_key: str,
    out_path: Path,
    title_prefix: str,
) -> None:
    """Draw 4-slice grids (every 10 rounds from 0–40) with x=utterance, y=meaning."""
    round_index = _build_round_index(history)
    slice_rounds = _slice_rounds(history, step=10, max_round=40, max_slices=4)
    if not slice_rounds:
        return

    meanings = sorted({m for entry in history for m in entry[agent_key]["meaning_to_utterance"].keys()})
    utterances_raw = sorted(
        {
            u
            for entry in history
            for d in entry[agent_key]["meaning_to_utterance"].values()
            for u in d.keys()
        }
    )
    u_to_m = _infer_utterance_meaning(history, agent_key)
    meaning_order = {m: idx for idx, m in enumerate(meanings)}
    utterances = sorted(
        utterances_raw,
        key=lambda u: (meaning_order.get(u_to_m.get(u, ""), 1e9), u),
    )
    utterances = list(reversed(utterances))
    if not meanings or not utterances:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), squeeze=False)
    vmax = 1.0
    vmin = 0.0
    for idx, r in enumerate(slice_rounds):
        ax = axes[idx // 2][idx % 2]
        entry = round_index.get(r)
        if not entry:
            ax.axis("off")
            continue
        dist_m2u = entry[agent_key]["meaning_to_utterance"]
        matrix = np.array(
            [[_normalize(dist_m2u.get(m, {})).get(u, 0.0) for u in utterances] for m in meanings]
        )
        im = ax.imshow(matrix, aspect="equal", cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(f"Round {r}")
        ax.set_xticks(range(len(utterances)))
        ax.set_xticklabels([_shorten_from_meaning(u_to_m.get(u, u)) for u in utterances], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(meanings)))
        ax.set_yticklabels(meanings, fontsize=8)
        ax.set_xlabel("Utterance")
        ax.set_ylabel("Meaning")

    # Hide unused subplots
    for j in range(len(slice_rounds), 4):
        axes[j // 2][j % 2].axis("off")

    fig.suptitle(f"{title_prefix}: P(utterance|meaning) slices (every 10 rounds, 0–40)", fontsize=14)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.18, location="right")
    cbar.set_label("Probability")
    fig.subplots_adjust(top=0.9, right=0.82, hspace=0.35, wspace=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _utterance_to_meaning_slice_grids(history: List[Dict], out_path: Path) -> None:
    """Builder u->m slices derived from builder meaning->utterance; 4 slices every 10 rounds (0–40)."""
    round_index = _build_round_index(history)
    slice_rounds = _slice_rounds(history, step=5, max_round=20, max_slices=4)
    if not slice_rounds:
        return

    # Collect global vocab
    meanings = sorted({m for entry in history for m in entry["builder"]["meaning_to_utterance"].keys()})
    utterances_raw = sorted(
        {
            u
            for entry in history
            for d in entry["builder"]["meaning_to_utterance"].values()
            for u in d.keys()
        }
    )
    u_to_m = _infer_utterance_meaning(history, agent_key="builder")
    meaning_order = {m: idx for idx, m in enumerate(meanings)}
    utterances = sorted(
        utterances_raw,
        key=lambda u: (meaning_order.get(u_to_m.get(u, ""), 1e9), u),
    )
    utterances = list(reversed(utterances))
    if not meanings or not utterances:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), squeeze=False)
    vmax = 1.0
    vmin = 0.0

    for idx, r in enumerate(slice_rounds):
        ax = axes[idx // 2][idx % 2]
        entry = round_index.get(r)
        if not entry:
            ax.axis("off")
            continue
        m2u = entry["builder"]["meaning_to_utterance"]
        matrix_rows: List[List[float]] = []
        for m in meanings:
            # Build u->m by normalizing column-wise over meanings
            row: List[float] = []
            for u in utterances:
                col_raw = {mm: dist.get(u, 0.0) for mm, dist in m2u.items()}
                col_norm = _normalize(col_raw)
                row.append(col_norm.get(m, 0.0))
            matrix_rows.append(row)
        matrix = np.array(matrix_rows)
        im = ax.imshow(matrix, aspect="equal", cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(f"Round {r}")
        ax.set_xticks(range(len(utterances)))
        ax.set_xticklabels([_shorten_from_meaning(u_to_m.get(u, u)) for u in utterances], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(meanings)))
        ax.set_yticklabels(meanings, fontsize=8)
        ax.set_xlabel("Utterance")
        ax.set_ylabel("Meaning")

    for j in range(len(slice_rounds), 4):
        axes[j // 2][j % 2].axis("off")

    fig.suptitle("Builder: P(meaning|utterance) slices (every 10 rounds, 0–40)", fontsize=14)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.12, location="right")
    cbar.set_label("Probability")
    fig.subplots_adjust(top=0.9, right=0.82, hspace=0.35, wspace=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def architect_task_confidence_heatmap(history: List[Dict], out_path: Path) -> None:
    """
    Grid of 6 subplots: one per task, rounds 0–50, y=programs (sorted by length asc), heat=value=P(program|task).
    """
    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)
    task_lengths = {
        tid: min(program_length(p) for p in info["correct_programs"]) if info.get("correct_programs") else 0
        for tid, info in towers_cfg.items()
    }
    tasks_sorted = [tid for tid, _ in sorted(task_lengths.items(), key=lambda kv: kv[1])]

    round_index = _build_round_index(history)
    max_round = max(round_index.keys()) if round_index else 0
    rounds = list(range(0, max_round + 1))

    fig, axes = plt.subplots(3, 2, figsize=(30, 20), squeeze=False)
    vmax = 1.0
    vmin = 0.0

    for idx, tid in enumerate(tasks_sorted):
        if idx >= 6:
            break
        ax = axes[idx // 2][idx % 2]
        programs = sorted(
            towers_cfg.get(tid, {}).get("correct_programs", []),
            key=lambda p: program_length(p),
        )
        if not programs:
            ax.axis("off")
            continue
        matrix_rows: List[List[float]] = []
        for p in programs:
            row: List[float] = []
            for r in rounds:
                entry = round_index.get(r)
                if not entry:
                    row.append(0.0)
                    continue
                dist = _normalize(entry["architect"]["task_to_program"].get(tid, {}))
                row.append(dist.get(p, 0.0))
            matrix_rows.append(row)
        matrix = np.array(matrix_rows)
        im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"Task {tid} (len↑)")
        ax.set_yticks(range(len(programs)))
        ax.set_yticklabels([f"{p} (L={program_length(p)})" for p in programs], fontsize=8)
        step = max(1, len(rounds) // 8)
        ticks = list(range(0, len(rounds), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([rounds[i] for i in ticks], rotation=45, ha="right")
        ax.set_xlabel(f"Round (0–{max_round})")
        ax.set_ylabel("Program")

    for j in range(len(tasks_sorted), 6):
        axes[j // 2][j % 2].axis("off")

    fig.suptitle("Architect: P(program|task) over rounds (per task, programs sorted by length)", fontsize=14)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.12, location="right")
    cbar.set_label("Probability")
    fig.subplots_adjust(top=0.9, right=0.85, hspace=0.5, wspace=0.5)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize belief_history.json as heatmaps.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "exp1" / "belief_history.json",
        help="Path to belief_history.json",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("plots"), help="Directory to save figures")
    args = parser.parse_args()

    out_dir: Path = args.out_dir / "exp1"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(args.input)
    if not history:
        print("No history entries found.")
        return

    _meaning_utterance_slice_grids(history, agent_key="architect", out_path=out_dir / "architect_m2u_heatmap.png", title_prefix="Architect")
    _utterance_to_meaning_slice_grids(history, out_path=out_dir / "builder_u2m_heatmap.png")
    architect_task_confidence_heatmap(history, out_path=out_dir / "architect_task_heatmap.png")
    print(f"Saved heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
