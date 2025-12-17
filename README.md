# Convention Formation and Transmission.

This repo implements a minimal cultural evolution sandbox inspired by [Compositional Abstractions Tutorial](https://github.com/cogtoolslab/compositional-abstractions-tutorial).

## Requirements
- Python 3.10+ (should work on 3.9+).
- Dependencies:
 `python -m pip install -r requirements.txt`


## Project Layout

- `data/task/task.json` — tower tasks and their correct programs. Extracted from the original project above.
- `data/task/lexicon.json` — lexicon dict includes meaning, utterance and the associated program. Extracted from the original project above.

- `config.py` — hyperparameters and paths.
- `task.py`, `lexicon.py` — task/lexicon  utilities.
- `agents.py` — `DyadAgent` class.
- `population.py` — population class.
- `exp1_dyad_convention.py` — single-dyad demo (round-level logs + curves).
- `plot.py` — heatmaps for Exp1.
- `exp2_population_convention.py` — paired vs mixed training mode for population.
- `plot2.py` — heatmaps for Exp2.
- `exp3_generation_convention.py` — multi-generation turnover experiment.
- `plot3.py` — heatmaps for Exp3.

Outputs:
- Plots → `plots/`
- JSON logs → `results/`

# Usage

## Exp1: Running a Single Dyad
```bash
python exp1_dyad_convention.py
python plot.py
```
Outputs:
- Plots under `plots/exp1/`
- JSON logs under `results/exp1/` 


## Exp2: Population Experiments(Paired vs Mixed)
```bash
python exp2_population_convention.py
python plot2.py
```
Outputs:
- Plots under `plots/exp2/`
- JSON logs under `results/exp2/` 

## Exp3: Turnover Across Generations
```bash
python exp3_generation_convention.py
python plot3.py
```
Outputs:
- Plots under `plots/exp3/`
- JSON logs under `results/exp3/` 


## Configuration
All global knobs live in `config.py`. 

| Parameter | What it controls (code) | If increased… | If decreased… |
|---|---|---|---|
| `LENGTH_PRIOR_LAMBDA` | Length bias (`lexicon.py`, `agents.py`) | **Note**: in `lexicon.py` the prior uses `exp(-λ*(L-1))` (λ↑ favors *shorter* meanings/programs), but in `agents.py` the task→program initialization uses `exp(+λ*(L-1))` (λ↑ favors *longer* programs). | Closer to uniform length preference (length trends become more data-driven). |
| `FIDELITY` | Strength of aligned meaning↔utterance pairs in the prior (`lexicon.py`) | More aligned prior → often higher early accuracy and less exploration. | Flatter prior → more exploration and slower convergence. |
| `ALPHA_SPEAKER` | Rationality in program choice (`agents.py: sample_program_for_task`) | Sharper softmax and more greedy. | More stochastic exploration. |
| `TASK_COST_WEIGHT` | Weight of program-length cost β (`agents.py: sample_program_for_task`) | Stronger preference for short programs. | More driven by belief probabilities. |
| `BELIF_UPDATE` | Belief update step size (`agents.py: update_*`) | More aggressive updates and faster changes. | More conservative updates. |
| `CHOICE_TEMPERATURE` | Sampling temperature τ (`agents.py: _apply_temperature`) | More random sample. | Closer to argmax. |

### Experiment defaults 
| Parameter | What it controls |
|---|---|
| `ROUNDS_PER_DYAD` | Rounds per generation. |
| `N_DYADS` | Number of dyads in the population (Exp2/Exp3). |
| `NEWPOP_RATIO` | Fraction of dyads replaced between generations (Exp3). |
| `GEN_POP` | Default number of generations (Exp3 default). | 
| `GLOBAL_SEED` | Random seed. |
| `EPSILON` | Numerical stability constant for safety.|