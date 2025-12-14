from typing import Dict, List

from interaction import run_dyad
from lexicon import LexiconEntry


def run_population(
    n_dyads: int,
    entries: List[Dict],
    lexicon_prior: Dict[str, Dict[str, float]],
    towers_cfg: Dict[str, Dict],
    n_rounds_per_dyad: int,
) -> Dict:

    from agents import AgentConfig, DyadAgent
    pop_results = []

    for i in range(n_dyads):
        cfg_A = AgentConfig(seed=100 + i * 2)
        cfg_B = AgentConfig(seed=101 + i * 2)
        agent_A = DyadAgent(entries, lexicon_prior, towers_cfg, cfg_A)
        agent_B = DyadAgent(entries, lexicon_prior, towers_cfg, cfg_B)
        pop_results.append({"dyad_id": i, "architect": agent_A, "builder": agent_B})

    return {"population": pop_results}
