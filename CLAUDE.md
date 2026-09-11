# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TLCS is a Deep Q-Learning agent that controls a single 4-way intersection in the SUMO traffic simulator. The agent learns to minimise cumulative vehicle waiting time by choosing one of 4 traffic-light phases at each decision step.

## Commands

All commands must be run from the **project root** (`refactor/`) so that relative paths in `constants.py` and the SUMO config resolve correctly.

```bash
# Install (runtime + dev tools)
pip install -e ".[dev]"

# Train (CLI script)
python src/train.py
python src/train.py --settings settings/training_settings.yaml --out model/

# Lint
ruff check src/

# Type-check
mypy src/

# Format
ruff format src/

# Test
pytest
```

Unit tests cover the shared metric definitions (`tests/test_metrics.py`); they
need neither SUMO nor a trained model. Everything else is verified by running.

The import smoke-test from the root:

```bash
cd src && python -c "
from environment import Environment, EnvStats
from agent import Agent, Memory, Sample
from policy import EpsilonGreedyPolicy
from episode import Record, run_episode
from settings import load_training_settings, load_testing_settings
from metrics import total_delay, avg_queue, sum_negative_rewards
from plots import save_data_and_plot
print('All imports OK')
"
```

## Architecture

The codebase separates the three standard RL components into distinct sub-packages under `src/`:

```
src/
├── constants.py          # All magic numbers, lane geometry, TL phase IDs, paths
├── metrics.py            # Episode metric definitions — shared by train/test/grid/baseline
├── settings.py           # Pydantic models (TrainingSettings, TestingSettings) + YAML loaders
├── logger.py             # RichHandler setup; configures root logger at import time
├── episode.py            # run_episode() — wires env + agent for one episode
├── train.py              # training_session() + __main__ entry point
├── plots.py              # save_data_and_plot()
│
├── environment/          # MDP environment (owns SUMO)
│   ├── core.py           # Environment class + EnvStats dataclass
│   ├── generator.py      # Weibull route-file generation
│   ├── state.py          # get_state(), get_lane_cell() — pure functions over TraCI
│   └── reward.py         # get_cumulated_waiting_time(), get_queue_length() — pure functions
│
├── agent/                # Learning algorithm (owns Q-network + replay buffer)
│   ├── agent.py          # Agent class — Q-learning update (replay())
│   ├── model.py          # MLP nn.Module + Model wrapper (predict/train/save/load)
│   └── memory.py         # Memory replay buffer + Sample dataclass
│
└── policy/               # Action selection (separate from learning)
    └── epsilon_greedy.py # EpsilonGreedyPolicy — select_action(), set_epsilon()
```

**Dependency order** (no circular imports):
`constants` → `logger` → `metrics` → `model` → `policy` → `memory` → `agent` → `environment` → `episode` → `train`

(`metrics` depends on nothing — not on the project, not on TraCI. That is what lets
`comparison/` import it and what makes it testable without SUMO.)

**Key design decisions:**
- `metrics.py` is the single definition of every episode-level number that gets
  reported. `train.py`, `test.py`, `grid_train.py` and `comparison/run_baseline.py`
  all import it, so the agent and the fixed-time baseline are scored by the same
  function. Do not add a fourth aggregation — that is exactly how the units came
  apart before (PROBLEMS.md P0-1/P0-2). `total_delay` is vehicle-seconds; the
  negative-reward sum is a *training diagnostic* and must never be plotted against
  a baseline.
- `policy/` is intentionally separate from `agent/` so the exploration rule can be swapped without touching the Q-learning update.
- `environment/state.py` and `environment/reward.py` are pure functions (no instance state) — they call TraCI directly, making them testable without a live SUMO process.
- `agent/model.py` saves/loads via `state_dict` (not pickle). Saved `.pt` files from the original project are **not compatible**.
- All path constants are relative to the project root. Scripts/notebooks must `os.chdir(PROJECT_ROOT)` or be launched from `refactor/` before SUMO starts.

## Runtime Artifacts

| Path | Created by | Notes |
|------|-----------|-------|
| `intersection/episode_routes.rou.xml` | `generator.py` at episode start | Do not commit |
| `model/trained_model.pt` | `train.py` after training | Weights only (state_dict) |
| `model/plot_*.png` / `model/plot_*_data.txt` | `plots.py` after training | `delay` (vehicle-seconds) and `queue` are comparable to the baseline; `reward` is a training diagnostic only |
| `comparison/archive/` | — | Results from before the metric fix; `compare_results.py` refuses them |

## Configuration

Edit `settings/training_settings.yaml` to change hyperparameters. The `sumocfg_file` path inside the YAML is relative to the project root. SUMO must be installed and `SUMO_HOME` set in the environment before running.
