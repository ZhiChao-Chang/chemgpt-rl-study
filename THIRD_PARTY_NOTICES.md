# Third-party sources and rights

- **ChemGPT-4.7M** (`ncfrey/ChemGPT-4.7M`): pretrained model revision and SHA-256 identities are recorded with the proposal and experiment protocols. We do not include downloaded pretrained weights or claim that a code license automatically licenses the model files.
- **SELFIES 1.0.4 distribution**: historical decoder implementation is retained in `experiments/vendor/`; its original license is preserved in `selfies-1.0.4.dist-info/LICENSE`. The wheel's internal version string differs from the distribution version; provenance retains both facts.
- **RDKit**: used for molecular parsing, descriptors, fingerprints, drawing, shared-subgraph matching and illustrative conformer construction. It is an environment dependency, not a newly authored biological predictor.
- **GuacaMol**: benchmark/scoring source referenced by the frozen study implementation. The relevant source revision is documented in the experimental contract.
- **3Dmol.js**: the embedded renderer asset includes its original license text in `experiments/visual_report/src/content/assets/molecular_viewer_library.json`, also accessible from the report's structure-method details. Only the renderer was reused from the reference presentation; no antibody research data is used as RL evidence.
- **React, Vite and the browser report runtime**: runtime source and the locked npm dependency manifest are retained with the report. Dependency packages keep their respective licenses; no blanket relicensing is asserted.
- **Course materials**: downloaded lecture slides are not part of this repository upload.

This notice identifies third-party components; it does not grant rights beyond their respective terms or claim ownership of third-party research material.
