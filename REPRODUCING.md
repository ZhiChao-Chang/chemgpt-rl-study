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

The builder excludes local conversation metadata from the published page. `docs/publication-manifest.json` binds the HTML SHA-256 to its evidence cutoff and run count. The repository check verifies result hashes and recomputes grouped means and sample standard deviations without altering original ledgers.

## Updating reviewed results

1. Preserve the original ledgers and record their source identities.
2. Rebuild the snapshot with `experiments/visualization_work/prepare_snapshot.py` in a compatible RDKit environment. This regenerates molecular assets; follow it with `build_conformers.py` and `redesign_evidence.py` if refreshing all current structure comparisons.
3. Reconcile the authored narrative, counts and evidence cutoff with the same results; do not update numbers while retaining unsupported prose.
4. Build, verify and review changed controls/exports at desktop and mobile widths.
5. Commit source, data and `docs/` together. Static hosting must serve only the selected presentation artifact, not the research repository root.

These are maintenance steps, not automatic permission to launch a new experiment or GPU job.

## Hosting

The source repository is private. As of setup, its account's GitHub Pages settings require an upgrade or a public repository. Neither an account purchase nor changing project visibility is authorized by the current setup. The static artifact is ready for a separately approved host or a presentation-only public repository; the research source can remain private.

The Report build workflow builds and verifies on GitHub, then uploads the static presentation as an authenticated Actions artifact. It is a build artifact, not an already deployed online site.

## Historical experimental reproducibility

The core and extension implementations, configurations, ledgers and analyses are versioned in their original paths. Their recorded model revision, hashes and software metadata describe the actual experiments. Large model files and checkpoints are deliberately excluded from Git and stay untouched locally.

The original `REPORT.md`, `selection/` and proposal-stage reports describe earlier stages. Do not treat their old “not yet trained” statuses as the current study status; current completed results are the 126 runs referenced by `experiments/visual_report/src/data.json`.
