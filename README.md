# Convention Evolution (Builder–Architect RSA Simulation)

This repo implements a minimal cultural evolution sandbox: two RSA-based agents (Architect ↔ Builder) learn a shared lexicon that maps DSL programs (meanings) to natural-language utterances while solving tower-building tasks. It includes single-dyad runs, population scaffolding, logging, and visualization.

## Project Layout
- `config.py` — hyperparameters and paths.
- `data/task/task.json` — tower tasks and their correct programs.
- `data/task/lexicon.json` — initial meaning–program–utterance entries.
- `agents.py` — `DyadAgent` with task→program and meaning↔utterance beliefs, sampling, and updates.
- `task.py`, `lexicon.py` — task/lexicon loaders and utilities.
- `test.py` — single-dyad demo, logging, and training curves.
- `plot.py` — heatmaps for belief trajectories from `belief_history.json`.
- `population.py`, `main.py` — population scaffold (multiple dyads; extend as needed).

Outputs:
- Plots → `plots/`
- JSON logs → `results/`

## Core Modeling Ideas
### Meaning and Programs
- Meanings are DSL programs (primitives `h, v, l, r` plus chunked shapes `chunk_8, chunk_C, chunk_L, chunk_Pi`).
- Tasks define sets of correct programs (`data/task/task.json`).

### Length-Based Prior (lexicon.py)
For a meaning $m$ with length $L(m)$ (number of actions):
$$
w(m) = \exp(-\lambda \cdot (L(m) - 1)), \quad P(m) = \frac{w(m)}{\sum_{m'} w(m')}.
$$
`build_entries_with_prior` expands the lexicon and produces a matrix prior $P(u \mid m)$ by boosting aligned utterances with `FIDELITY`. Concretely, with length prior $P(m)$ and an utterance’s canonical meaning $m^*(u)$:
$$
P(u \mid m) \propto
\begin{cases}
\text{FIDELITY} \cdot P(m^*(u)) & m = m^*(u), \\
P(m^*(u)) & \text{otherwise},
\end{cases}
$$
then normalized over $u$ for each $m$. This prior seeds agent beliefs.

### Agent Beliefs (agents.py)
- Meaning↔Utterance beliefs: `belief_m2u[m][u]`, `belief_u2m[u][m]`, initialized from the prior matrix.
- Task→Program belief: for each task, a length-aware prior over candidate programs.
- Meaning prior for pragmatic listening is marginalized from `belief_m2u`.

### Utilities and Sampling
- Task→Program sampling:
  - Utility per program $p$ (for task $t$):  
    $$
    U(p) = (1-\beta_t)\log P(p \mid t) - \beta_t \log \text{cost}(p),
    $$
    scaled by `alpha_speaker`; softmaxed to sample.
  - `cost(p)` is program length.
- Speaker $P(u \mid m)$: uses literal listener $L_0(m \mid u)$ → utility $\log L_0(m \mid u)$ with cost weight `beta_u`; softmax with `alpha_speaker`.
- Pragmatic listener $P(m \mid u)$: $S_1(u \mid m) \times P(m)$, normalized.
- Updates:
  - Task belief updated proportionally to fraction of correctly decoded actions.
  - Meaning↔utterance beliefs updated only when the listener’s guess matches the gold meaning for that utterance.

## Running a Single Dyad
```bash
python3 test.py \
  --rounds 50 \
  --lambda-length 0.3 \
  --seed 42 \
  --log-prefix sampling_log
```
Outputs:
- `plots/loss_curve.png`, `plots/accuracy_curve.png`, `plots/program_length_curve.png`
- `results/belief_history.json` (full belief trace)
- `results/sampling_log_architect.json`, `results/sampling_log_builder.json` (per-sample distributions)

## Visualizing Belief Trajectories
After running `test.py` (which writes `results/belief_history.json`):
```bash
python3 plot.py --input results/belief_history.json --out-dir plots
```
Generates:
- `architect_m2u_slices.png` — 4 slices (rounds 0–20, step 5) of $P(u \mid m)$ for the architect.
- `builder_u2m_slices.png` — 4 slices of derived $P(m \mid u)$ for the builder.
- `architect_task_heatmap.png` — per-task grids of $P(\text{program} \mid \text{task})$ over rounds, programs sorted by length.

## Population Scaffold
`main.py` and `population.py` illustrate how to spin up multiple dyads. The current scaffold initializes independent dyads; extend `population.py` to aggregate accuracies, share lexica, or model cultural transmission.

## Configuration
Key knobs in `config.py`:
- `LENGTH_PRIOR_LAMBDA` — length prior λ.
- `FIDELITY` — boost for aligned meaning–utterance pairs in the prior.
- `ALPHA_SPEAKER` — speaker rationality (sharpens softmax).
- `UTTERANCE_COST_WEIGHT`, `TASK_COST_WEIGHT` — cost weights for speaker and task sampling.
- `BELIF_UPDATE` — belief update step size.
- `GLOBAL_SEED` — default RNG seed.

## Data Files
- `data/task/task.json` — tasks and candidate programs.
- `data/task/lexicon.json` — meaning↔utterance seed entries; modify to experiment with new templates or chunks.

## Notes and Tips
- All plots/JOSN outputs are organized under `plots/` and `results/`.
- Extend `plot.py` to add more diagnostics (e.g., KL to prior, entropy trajectories).
