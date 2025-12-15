from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from config import *
from lexicon import build_entries_with_prior, load_lexicon_config
from population_core import (
    Dyad,
    Population,
    build_population,
    pack_population,
    run_population,
    sample_dyads,
)
from task import load_towers_config

__all__ = [
    "Dyad",
    "Population",
    "sample_dyads",
    "pack_population",
    "build_population",
    "run_population",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Population evolution demo based on DyadAgents.")
    parser.add_argument("--dyads", type=int, default=N_DYADS, help="Number of dyads (speaker-listener pairs).")
    parser.add_argument(
        "--rounds-per-dyad",
        type=int,
        default=ROUNDS_PER_DYAD,
        help="Training rounds per dyad in each generation.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="How many generations to simulate for population evolution.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paired",
        choices=["paired", "mixed"],
        help="Training mode: paired (each dyad self-trains) or mixed (cross-dyad speaker/listener).",
    )
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed for reproducibility.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional directory to store per-agent sampling logs.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to store generation summaries and dyad belief snapshots.",
    )
    args = parser.parse_args()

    lex_cfg = load_lexicon_config(LEXICON_CONFIG_PATH)
    entries, lexicon_prior, meaning_prior = build_entries_with_prior(lex_cfg, lam=LENGTH_PRIOR_LAMBDA)
    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)

    results = run_population(
        n_dyads=args.dyads,
        entries=entries,
        lexicon_prior=lexicon_prior,
        meaning_prior=meaning_prior,
        towers_cfg=towers_cfg,
        n_rounds_per_dyad=args.rounds_per_dyad,
        n_generations=args.generations,
        seed=args.seed,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        training_mode=args.mode,
    )

    results_dir_path = Path(args.results_dir)
    print(f"Saved population summary to {results_dir_path / 'population_results.json'}")
    print(f"Saved dyad belief snapshots to {results_dir_path / 'dyad_beliefs.json'}")

    last_gen: Dict[str, Any] = results["generations"][-1] if results["generations"] else {}
    conv_stats: Dict[str, Any] = last_gen.get("evolute_convention", {})
    mean_acc = last_gen.get("mean_accuracy", 0.0)
    overall_js = conv_stats.get("overall_js_divergence", 0.0)
    print(f"Simulated {args.generations} generations × {args.dyads} dyads.")
    print(f"Final generation mean accuracy: {mean_acc:.3f}, overall JS divergence: {overall_js:.3f}")
    for dyad in last_gen.get("dyads", []):
        print(
            f"  Dyad {dyad['dyad_id']:02d} "
            f"({dyad['speaker_role']}→{dyad['listener_role']}): "
            f"acc={dyad['accuracy']:.3f}, loss={dyad['avg_loss']:.3f}"
        )


if __name__ == "__main__":
    main()
