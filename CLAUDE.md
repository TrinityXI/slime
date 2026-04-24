# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**slime** is an LLM post-training framework for RL scaling. It connects **Megatron** (training) with **SGLang** (fast inference/rollout) via **Ray** (distributed orchestration). The primary training loops are in `train.py` (synchronous) and `train_async.py` (asynchronous).

## Commands

### Setup
```bash
pip install -e . --no-deps
```

### Linting & Formatting
```bash
pre-commit run --all-files --show-diff-on-failure --color=always
```
Hooks: `ruff --fix`, `autoflake`, `isort`, `black` (119-char line length).

### Running Tests
```bash
pytest tests/                                 # all tests
pytest tests/ -m "unit"                       # unit tests only
pytest tests/test_qwen3_4B_ppo.py -vv        # single test file
pytest --durations=0 --strict-markers -vv    # verbose with timing
```

For GPU-requiring tests (as used in CI):
```bash
python tests/ci/gpu_lock_exec.py --count 8 -- python tests/test_qwen3_4B_ppo.py
```

Test markers: `unit`, `integration`, `system`, `acceptance`, `docs`, `skipduringci`, `pleasefixme`.

## Architecture

### Three Core Modules

```
Prompts (JSONL/Parquet)
    ↓
Data Buffer (slime/utils/data.py)
    ↓
Rollout Manager (slime/ray/rollout.py)
    → SGLang Engines → generate responses + rewards
    → Custom rollout fn, reward models, dynamic filters
    ↓
Training Actor Group (slime/ray/actor_group.py + train_actor.py)
    → Megatron (slime/backends/megatron_utils/)
    → PPO loss (slime/utils/ppo_utils.py + mask_utils.py)
    → Sync params back to rollout engines
    ↓
Checkpoints + Metrics (W&B / Tensorboard)
```

1. **Training** — `slime/backends/megatron_utils/`: actor forward/backward, loss computation, checkpoint management, data pipeline integration.
2. **Rollout** — `slime/rollout/` + `slime/backends/sglang_utils/`: SGLang engine management, reward models (`rm_hub/`), dynamic filters (`filter_hub/`).
3. **Data Buffer** — `slime/utils/data.py`: bridges prompts (JSONL/Parquet) with rollout/training.

### Ray Orchestration (`slime/ray/`)

- `placement_group.py` — allocates GPU resources across nodes
- `rollout.py` — `RolloutManager` Ray actor; owns SGLang engines and orchestrates data generation
- `actor_group.py` — manages the distributed Megatron training actor fleet
- `train_actor.py` — individual training actor implementation

### Configuration (`slime/utils/arguments.py`)

All CLI arguments flow through `get_slime_extra_args_provider()`, organized into groups: cluster (node/GPU layout), train (LR, batch size, loss masking), rollout (data generation settings), data (prompt paths), evaluation, algorithm (PPO params), fault tolerance, router, W&B/Tensorboard, and debug flags.

Arguments fall into three namespaces:
- **Megatron** — standard Megatron-LM flags (e.g., `--tensor-model-parallel-size`)
- **SGLang** — prefixed with `--sglang-` (e.g., `--sglang-mem-fraction-static`)
- **slime-specific** — framework-level flags defined in `arguments.py`

### Extension Points

Custom logic is injected via path arguments; each loads a Python module at runtime:

| Argument | Purpose |
|---|---|
| `--rollout-function-path` | Custom rollout/data generation logic |
| `--custom-rm-path` | Custom reward model |
| `--custom-model-provider-path` | Custom model architecture |
| `--log-rollout-data-path` | Custom rollout data logging |
| `--eval-config` | Per-dataset evaluation overrides |

Plugin implementations live in `slime_plugins/` and `examples/`.

### Data Processing (`data_processing/infialign/`)

Post-training data cleaning pipeline with two main components:

**Deduplication** (`deduplication/`):
- **Sample-level** — removes duplicate query+response pairs via exact matching + MinHash/LSH/WCC (Spark, threshold 0.85)
- **Query-level Stage 1** — deduplicates queries while aggregating all responses, using exact + MinHash/LSH/WCC (Spark)
- **Query-level Stage 2** — semantic deduplication via SentenceTransformer (BGE-M3) embeddings + FAISS similarity search (threshold 0.9, GPU-accelerated)

**Decontamination** (`decontamination/`):
- Removes evaluation benchmark contamination via word-level N-gram matching (size 32)
- Benchmarks tracked in `benchmarks.yaml`: MMLU-Pro, MMLU, SuperGPQA, AIME, MATH-500, GSM8K, GPQA-Diamond, etc.

Each component follows: `config.sh` (params) → `dedup.py`/`decontamination.py` (logic) → `run_*.sh` (Spark setup) → `submit.sh` (SLURM submission, min 2 nodes).

## Custom Skills

The `.claude/skills/` directory contains task-specific guides. Use the Skill tool to invoke them when relevant:

- `/add-rollout-function` — wire in a custom rollout function
- `/add-reward-function` — add a custom reward model
- `/add-eval-dataset-config` — configure evaluation datasets
- `/add-dynamic-filter` — add per-sample filtering/masking hooks
- `/add-tests-and-ci` — add tests and register them in CI

## CI

CI triggers via PR labels (e.g., `run-ci-short`, `run-ci-megatron`). The `e2e-test-plugin-contracts` job runs on CPU; all other e2e jobs require GPUs and use `tests/ci/gpu_lock_exec.py` to serialize GPU access.
