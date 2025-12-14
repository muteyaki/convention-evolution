# main.py
from pathlib import Path

from config import (
    LEXICON_CONFIG_PATH,
    TOWERS_CONFIG_PATH,
    N_DYADS,
    ROUNDS_PER_DYAD,
)
from lexicon import (
    load_lexicon_config,
    build_entries_with_prior,
)
from task import load_towers_config
from agents import AgentConfig, DyadAgent
from population import run_population

def main():
    # 1) 读入 lexicon 和 towers 配置
    lex_cfg = load_lexicon_config(LEXICON_CONFIG_PATH)
    entries, lexicon_prior = build_entries_with_prior(lex_cfg)
    towers_cfg = load_towers_config(TOWERS_CONFIG_PATH)

    # 2) 跑一个 population-level 模拟
    results = run_population(
        n_dyads=N_DYADS,
        entries=entries,
        lexicon_prior=lexicon_prior,
        towers_cfg=towers_cfg,
        n_rounds_per_dyad=ROUNDS_PER_DYAD,
    )

    # 简单打印一下结果概览
    accuracies = [d["accuracy"] for d in results["population"]]
    avg_acc = sum(accuracies) / len(accuracies)
    print(f"Ran {N_DYADS} dyads × {ROUNDS_PER_DYAD} rounds")
    print("Accuracies:", accuracies)
    print("Avg accuracy:", avg_acc)

if __name__ == "__main__":
    main()
