"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LEXICON_CONFIG_PATH = DATA_DIR / "task" / "lexicon.json"
TOWERS_CONFIG_PATH = DATA_DIR / "task" / "task.json"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

LENGTH_PRIOR_LAMBDA = 0.4  # Length prior λ.
FIDELITY = 1.5
ALPHA_SPEAKER = 1.5
TASK_COST_WEIGHT = 0.8
BELIF_UPDATE = 1
CHOICE_TEMPERATURE = 0.5

N_DYADS = 16
ROUNDS_PER_DYAD = 20
NEWPOP_RATIO = 0.5
GEN_POP = 5

GLOBAL_SEED = 42
EPSILON = 1e-8
