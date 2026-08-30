# Released figures and results

This directory contains the compact snapshot used to produce the paper:

- `figures/` contains the 24 figure PDFs referenced by the manuscript;
- `results/en_ablation/` contains the main comparison and robustness tables;
- `results/target_language/` contains per-language summaries, axis results, and
  bootstrap intervals;
- `results/seed_controls/`, `results/multilingual/`,
  `results/worldvaluesbench/`, and `results/validation/` contain the associated
  controls and extensions.

`manifest.json` records each file's original generated path, byte size, and
SHA-256 digest. Refresh the snapshot from local outputs with:

```bash
python src/pipeline/export_release_artifacts.py
```

The release omits model checkpoints, cached representation arrays, and the
22 MB per-word neighbor table. All 40 public model IDs are listed in
`configs/models/models_hub.json`; all omitted results can be regenerated with
the entrypoints in `configs/test_manifest.json`.
