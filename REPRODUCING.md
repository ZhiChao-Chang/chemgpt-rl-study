# Reproduction and presentation deployment

## Read or build the report

`docs/index.html` is a portable browser artifact, not a link back to localhost. It embeds the current reviewed values and molecular comparisons. Serve `docs/` with any static host or open the file directly. Optional 3D requires WebGL; 2D and numerical views remain available without it.

To rebuild from this repository alone:

```sh
cd experiments/visual_report
npm ci
cd ../..
python3 tools/build_site.py
python3 tools/verify_repository.py
```

Use Node.js 24 and Python 3. No GPU, external model endpoint, Codex installation or remote training controller is required. `npm ci` uses the committed lockfile; the source runtime is verified against its scoped integrity manifest before Vite builds the single HTML.

Before rendering, `tools/prepare_report_data.py` reconstructs the numerical rows from the 126 curated runs in `data/study_manifest.json`. It checks 506 original files and 252 case rows, then emits per-run/grouped CSVs and a data receipt using only the Python standard library. Descriptor and diversity evidence is read with recorded hashes; drawings and conformers are separate precomputed assets.

The builder excludes local conversation metadata from the published page. `docs/publication-manifest.json` binds the HTML SHA-256 to its evidence cutoff and run count. The repository check verifies result hashes and recomputes grouped means and sample standard deviations without altering original ledgers.

## Updating reviewed results

1. Preserve original result/config/search/evaluation files and their identities.
2. Intentionally update `data/study_manifest.json` when changing the study population; scratch directories are not auto-selected.
3. Run `python3 tools/prepare_report_data.py` to rebuild browser numerical rows, per-run CSV, grouped CSV and its receipt from those records.
4. If molecular identities or selected cases change, regenerate chemistry-dependent assets with the scripts in `experiments/visualization_work/` in a compatible RDKit environment; keep provenance and explicit missing-conformer states.
5. Reconcile the authored findings, counts, selection definitions and evidence cutoff with the same study.
6. Build and verify, then commit the research inputs, data, source and offline artifact together.

These are maintenance steps, not permission to launch new training or GPU jobs.

## Hosting from the research repository

The user has authorized making this main research repository public and using GitHub Pages. GitHub currently requests the account owner's email confirmation to finish the visibility change. Once public, choose **Settings → Pages → Source → GitHub Actions** and run **Research pipeline and Pages** once. Subsequent `main` pushes rebuild and deploy automatically.

The workflow uses Node 24 and standard Python, installs the locked browser dependencies, reconstructs data from the source records, verifies the artifact and publishes **only `docs/`** to Pages. The project itself remains a normal code/data/analysis repository. The site URL is `https://zhichao-chang.github.io/chemgpt-rl-study/` after initial deployment succeeds.

The `public` event builds/verifies without attempting deployment until Pages is configured; private pushes likewise only build. No personal token, paid upgrade, GPU server or author's running computer is required. The deployed manifest records the workflow commit and the source-manifest hash.

## Historical experimental reproducibility

The core and extension implementations, configurations, ledgers and analyses are versioned in their original paths. Their recorded model revision, hashes and software metadata describe the actual experiments. Large model files and checkpoints are deliberately excluded from Git and stay untouched locally.

The original `REPORT.md`, `selection/` and proposal-stage reports describe earlier stages. Do not treat their old “not yet trained” statuses as the current study status; current completed results are the 126 runs referenced by `experiments/visual_report/src/data.json`.
