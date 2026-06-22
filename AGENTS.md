# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives under `src/`:

- `src/data/` handles clinical preprocessing, CT transforms, datasets, and loaders.
- `src/models/` defines CT and clinical encoders, fusion, and the top-level predictor.
- `src/training/` contains losses, metrics, and training loops.
- `src/utils/` provides configuration and experiment logging helpers.

Use `scripts/preprocess.py`, `scripts/train.py`, and `scripts/evaluate.py` as command-line entry points. Experiment settings belong in `configs/`; raw and derived datasets belong in `data/raw/` and `data/processed/`. Keep exploratory work in `notebooks/`, not in production modules.

## Setup, Training, and Development Commands

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Common workflows:

```bash
python scripts/preprocess.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/train.py --override training.batch_size=8
python scripts/evaluate.py --config configs/default.yaml \
  --checkpoint checkpoints/best_model.pt
python -m compileall src scripts
```

The final command is a lightweight syntax/import smoke check. Training outputs are written to `checkpoints/` and logging outputs to `runs/` by default.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, variables, modules, and YAML keys; use `PascalCase` for classes; use `UPPER_CASE` for constants. Add type hints and concise docstrings to public functions and classes. Prefer configuration-driven behavior over hard-coded paths or hyperparameters. Keep imports grouped as standard library, third-party, then local `src` imports.

## Testing Guidelines

No automated test suite is currently configured. Add new tests under `tests/`, mirroring the source layout, with names such as `tests/data/test_clinical_processor.py`. Use `pytest` for new tests and run `python -m pytest`; add it to development dependencies when introducing the first test. Cover preprocessing edge cases, tensor shapes, configuration overrides, metric calculations, and small CPU-only model passes. Never require private clinical data in tests—use synthetic fixtures.

## Commit & Pull Request Guidelines

Git history is not available in this checkout, so no repository-specific convention can be inferred. Use short, imperative commits such as `Add stratified patient split`. Keep each commit focused. Pull requests should explain the motivation, summarize changes, list validation commands, and note configuration or data-layout changes. Include metric comparisons for model changes and never commit patient data, checkpoints, credentials, or experiment logs.
