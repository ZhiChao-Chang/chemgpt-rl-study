# Study population and derived numerical data

The original experiment records are versioned under `experiments/collected_remote/results/` and `experiments/exploration_11h_20261114/collected/results/`. This directory defines the study population and contains reproducible numeric extracts, not a replacement for those records.

| File | Role |
| :--- | :--- |
| `study_manifest.json` | Explicit 126-run selection, cohort identity and SHA-256 for each result/config/search/evaluation file plus descriptor evidence. |
| `run_metrics.csv` | One row per training run; raw original reward, search metrics and documented property/diversity summaries. |
| `summary_metrics.csv` | One row per cohort/task/condition; equal-weight means and sample SD across 3 or 5 training seeds. |
| `report_data_receipt.json` | Data-build identities, source count, case-row checks and independent AUC reconstruction error. |

Rebuild these files and the website's numerical rows with:

```sh
python3 tools/prepare_report_data.py
```

This uses only the Python standard library and already computed records. The original descriptor/diversity evidence is read with its own hash; it is not silently recomputed with a potentially different chemistry library. Selected case rewards/components are checked against their original rows. Molecular drawings/conformers remain separate derived assets.

The manifest is an explicit study boundary. A new scratch result directory is not automatically absorbed into the published study; adding new experiments requires an intentional manifest and narrative update. Numerical consistency and exact replay do not establish biological validity.
