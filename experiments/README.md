# Research implementation and experiment records

This is the computational study, separate from its browser presentation. The experiments have already completed; reading records or rebuilding the site does not launch new training.

## Implementation

- `src/study.py`: fixed-prior scorer, on-policy sampling, REINFORCE / KL updates, unique-query accounting and independent terminal evaluation.
- `src/model_runtime.py`: model/token interface, probability masks and sequence-level calculations.
- `src/check_theory.py`: finite-MDP gradient and leave-one-out checks, biased-baseline negative control and KL chain-rule example.
- `exploration_11h_20261114/src/`: extension implementation and the frozen KL-dose, length/TPSA and long-budget schedule.
- `contract.json` and the extension contract: actual settings, populations, budgets and stopping rules.

## Recorded environment

The environment is recorded in `collected_remote/runtime_snapshot.json` and `exploration_11h_20261114/collected/results/environment.json`: Python 3.10.21, PyTorch 2.7.1+cu128, Transformers 4.51.3, Tokenizers 0.21.4, NumPy 1.26.4, RDKit 2025.9.6, SELFIES distribution 1.0.4. GPU runs used an NVIDIA H100 80GB HBM3. `requirements-research.txt` records these packages; it is not required by the browser-data build.

```sh
python3.10 -m venv .venv-research
. .venv-research/bin/activate
python -m pip install -r requirements-research.txt
python tools/fetch_model.py --destination .model-cache/ChemGPT-4.7M
```

Run from the repository root. The download command reads the exact revision and SHA-256 manifest; it does not train. Model weights are not distributed with the repository and retain their original rights. The SELFIES distribution's internal module version string differs from its package metadata; the source wheel and hash are documented.

## Technical checks without rerunning the campaign

To preserve the original theory receipt, run a copy:

```sh
CHEMGPT_CHECK_ROOT=$(mktemp -d)
mkdir -p "$CHEMGPT_CHECK_ROOT/src"
cp experiments/src/check_theory.py "$CHEMGPT_CHECK_ROOT/src/"
python "$CHEMGPT_CHECK_ROOT/src/check_theory.py"
```

A CPU model-interface gate uses an explicitly new output directory and stops before any training loop:

```sh
PYTHONPATH=experiments/vendor python experiments/src/study.py \
  --model-dir .model-cache/ChemGPT-4.7M \
  --contract experiments/contract.json \
  --output /tmp/chemgpt-interface-check-new \
  --device cpu --phase gate
```

The `dev` and `all` phases perform real optimization. The historical GPU program guards its H100 execution environment, and the extension contract contains an expired absolute deadline. These original files are preserved as-run; a new campaign requires a separate reviewed protocol, new output path and resource authorization rather than silently rewriting the archived experiment.

## Existing results and derived analysis

`collected_remote/results/main/` contains the 18 core runs. The extension's `collected/results/` contains 108 analytical runs, with development and exploratory checks kept separately. Each selected run retains result/config/search/evaluation files. `data/study_manifest.json` defines which 126 runs are used by the current analysis and records their file identities.

To inspect the completed research without model downloads:

```sh
python3 tools/prepare_report_data.py
python3 tools/verify_repository.py
```

The first reconstructs numeric data and checks case identities; the second verifies the packaged site and source records. See the root `PROJECT.md` for the full source map and `REPRODUCING.md` for website construction and Pages deployment.
