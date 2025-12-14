# config.py
from pathlib import Path

# ---- Path config ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LEXICON_CONFIG_PATH = DATA_DIR / "task" / "lexicon.json"
TOWERS_CONFIG_PATH = DATA_DIR / "task" / "task.json"

# ---- RSA / agent para ----
LENGTH_PRIOR_LAMBDA = 0.4        # length-based P(m) 的 λ
FIDELITY = 2.5
ALPHA_SPEAKER = 1.5              # speaker
TASK_COST_WEIGHT = 0.7      # cost(u)
BELIF_UPDATE = 2 # Dyad's update weight 

# ---- Population para ----
N_DYADS = 10
ROUNDS_PER_DYAD = 200

# System para
GLOBAL_SEED = 42
EPSILON = 1e-8
