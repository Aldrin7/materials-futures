# Notebook Cell Order — Updated with Final Refinements

After all experiments complete, insert these cells in order:

```
Cell 22:  Ablation summary + save (existing)
Cell 22b: Per-run configs + reproducibility artifacts (existing)
Cell 22c: Environment + git + data manifest + model export (existing)

NEW CELLS (insert here, before the existing Cell 23):
Cell 26b: Schema versioning + experiment index
Cell 26c: Automatic validation (pre-freeze)
Cell 26d: CITATION.cff
Cell 26e: Updated data manifest (full query details)

Continue with existing cells:
Cell 23:  Figure 1 — Parity plots (now also saves fig1_parity_data.csv)
Cell 24:  Figure 2 — Baseline comparison (now also saves fig2_comparison_data.csv)
...etc for all figures...

Cell 26:  Results summary + save
Cell 27:  Paper draft (auto-generated with actual numbers)
Cell 28:  Create README (use README_template.md content)
Cell 29:  Requirements.txt
Cell 30:  Results freeze + zip + upload
Cell 31:  Model loading utility
```

## Validation Flow

```
Training complete
  ↓
Save predictions (per experiment ID × split)
  ↓
Save figure data CSVs
  ↓
Build experiment index (experiment_index.csv)
  ↓
Run validation (Cell 26c)
  ↓
  ├── PASS → proceed to freeze
  └── FAIL → fix errors, re-run validation
  ↓
Freeze snapshot (FREEZE_MANIFEST.json)
  ↓
Write paper against frozen snapshot
```

## Schema Versions

All JSON artifacts now include:
```json
{
  "_schema": "run_config|validation_report|data_manifest|...",
  "_schema_version": "1.0",
  "_created": "2026-07-16T09:41:00Z",
  ...
}
```

When the pipeline evolves, bump `_schema_version` to "2.0".
Old experiments remain readable by version-aware parsers.
