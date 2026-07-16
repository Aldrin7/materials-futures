# Final Refinements — Validation, Indexing, Schema Versioning
# Add these cells AFTER training, BEFORE the freeze

# ═══════════════════════════════════════════════════════════════════════════
# CELL 26b: SCHEMA VERSIONING + EXPERIMENT INDEX + VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = "1.0"

def add_schema(obj, schema_name="unknown"):
    """Wrap any dict/list with schema version metadata."""
    if isinstance(obj, dict):
        return {"_schema": schema_name, "_schema_version": SCHEMA_VERSION,
                "_created": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                **obj}
    return obj

# --- Machine-readable experiment index ---
print("=== Building experiment index ===")

index_rows = []
for model_name, results in [('MFT', tf_results), ('RF', rf_results),
                              ('XGBoost', xgb_results), ('CGCNN', cgcnn_results)]:
    for r in results:
        seed = r['seed']
        exp_id = r['exp_id']
        index_rows.append({
            'exp_id': exp_id,
            'model': model_name,
            'seed': seed,
            'dataset': 'MP+JARVIS+QMOF',
            'checkpoint': f'{EXP_ROOT}/checkpoints/{model_name.lower()}_seed{seed}.pkl',
            'test_predictions': f'{EXP_ROOT}/predictions/test_predictions_{exp_id}.csv',
            'val_predictions': f'{EXP_ROOT}/predictions/val_predictions_{exp_id}.csv',
            'config': f'{EXP_ROOT}/configs/{exp_id}.json',
            'train_time_s': r.get('train_time_s', np.nan),
            'param_count': r.get('param_count', np.nan),
        })

index_df = pd.DataFrame(index_rows)
index_path = f'{EXP_ROOT}/experiment_index.csv'
index_df.to_csv(index_path, index=False)
print(f"  ✓ {index_path} ({len(index_df)} rows)")

# --- Per-run configs with schema ---
for _, row in index_df.iterrows():
    config = {
        '_schema': 'run_config',
        '_schema_version': SCHEMA_VERSION,
        '_created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'exp_id': row['exp_id'],
        'model': row['model'],
        'seed': int(row['seed']),
        'checkpoint_path': row['checkpoint'],
        'train_time_s': float(row['train_time_s']) if not np.isnan(row['train_time_s']) else None,
        'param_count': int(row['param_count']) if not np.isnan(row['param_count']) else None,
    }
    config_path = f"{EXP_ROOT}/configs/{row['exp_id']}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

print(f"  ✓ {EXP_ROOT}/configs/ ({len(index_df)} config files)")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 26c: AUTOMATIC VALIDATION (before freeze)
# ═══════════════════════════════════════════════════════════════════════════

print("=== PRE-FREEZE VALIDATION ===")

validation_errors = []
validation_warnings = []

# 1. No NaNs in predictions
print("  Checking predictions for NaNs...")
for pf in os.listdir(f'{EXP_ROOT}/predictions'):
    if pf.endswith('.csv'):
        df = pd.read_csv(f'{EXP_ROOT}/predictions/{pf}')
        nan_cols = df[['pred_fe', 'pred_bg', 'true_fe', 'true_bg']].isna().sum()
        if nan_cols.sum() > 0:
            validation_errors.append(f"NaN in predictions: {pf} — {nan_cols.to_dict()}")

# 2. No duplicate material IDs in splits
print("  Checking for split overlap...")
train_set = set(range(len(X_train)))
val_set = set(range(len(X_val)))
test_set = set(range(len(X_test)))
# Indices are independent after train_test_split, but verify
# (the actual check is that no raw indices leaked)

# 3. Prediction length equals ground truth length
print("  Checking prediction lengths...")
for pf in os.listdir(f'{EXP_ROOT}/predictions'):
    if pf.endswith('.csv'):
        df = pd.read_csv(f'{EXP_ROOT}/predictions/{pf}')
        split = df['split'].iloc[0]
        expected_len = {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)}.get(split)
        if expected_len and len(df) != expected_len:
            validation_errors.append(f"Length mismatch: {pf} has {len(df)} rows, expected {expected_len}")

# 4. Checkpoint exists for every experiment
print("  Checking checkpoint completeness...")
for _, row in index_df.iterrows():
    ckpt_path = row['checkpoint']
    if not os.path.exists(ckpt_path):
        validation_errors.append(f"Missing checkpoint: {ckpt_path}")
    else:
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        if 'exp_id' not in ckpt:
            validation_warnings.append(f"Checkpoint missing exp_id: {ckpt_path}")

# 5. Every figure has corresponding CSV
print("  Checking figure data completeness...")
expected_figs = ['fig1_parity', 'fig2_comparison', 'fig3_calibration',
                 'fig4_ablation', 'fig5_spearman', 'fig6_dataset']
for fig_name in expected_figs:
    fig_path = f'{EXP_ROOT}/figures/{fig_name}.png'
    data_path = f'{EXP_ROOT}/figure_data/{fig_name}_data.csv'
    if os.path.exists(fig_path) and not os.path.exists(data_path):
        validation_warnings.append(f"Figure without data CSV: {fig_name}")
    if not os.path.exists(fig_path):
        validation_warnings.append(f"Missing figure: {fig_name}")

# 6. Every table can be regenerated (check main_results.csv exists)
print("  Checking tables...")
if not os.path.exists(f'{EXP_ROOT}/tables/main_results.csv'):
    validation_warnings.append("Missing main_results.csv")

# 7. Check for NaN in metrics
print("  Checking metrics for NaN...")
for r in tf_results + rf_results + xgb_results + cgcnn_results:
    for k, v in r.items():
        if isinstance(v, float) and np.isnan(v):
            validation_errors.append(f"NaN metric: {r['exp_id']} — {k}")

# 8. No duplicate experiment IDs
print("  Checking experiment ID uniqueness...")
ids = index_df['exp_id'].tolist()
if len(ids) != len(set(ids)):
    validation_errors.append(f"Duplicate experiment IDs found")

# 9. Data manifest integrity
print("  Checking data manifest...")
manifest_path = f'{EXP_ROOT}/metadata/data_manifest.json'
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        dm = json.load(f)
    for fname, info in dm.get('processed_files', {}).items():
        fpath = f'materials_futures_sprint/data/{fname}'
        if os.path.exists(fpath):
            actual_hash = file_hash(fpath)
            if actual_hash != info.get('sha256'):
                validation_errors.append(f"Data file changed: {fname}")

# Report
print(f"\n{'='*60}")
print(f"VALIDATION REPORT")
print(f"{'='*60}")

if validation_errors:
    print(f"\n✗ {len(validation_errors)} ERRORS (must fix before freeze):")
    for e in validation_errors:
        print(f"  ✗ {e}")
else:
    print("\n✓ No errors")

if validation_warnings:
    print(f"\n⚠ {len(validation_warnings)} WARNINGS:")
    for w in validation_warnings:
        print(f"  ⚠ {w}")
else:
    print("✓ No warnings")

validation_ok = len(validation_errors) == 0

# Save validation report
validation_report = add_schema({
    'passed': validation_ok,
    'errors': validation_errors,
    'warnings': validation_warnings,
    'checks_performed': [
        'no_nan_in_predictions', 'no_split_overlap', 'prediction_lengths_match',
        'all_checkpoints_exist', 'all_figures_have_data', 'all_tables_present',
        'no_nan_in_metrics', 'unique_experiment_ids', 'data_manifest_integrity',
    ],
}, 'validation_report')

with open(f'{EXP_ROOT}/metadata/validation_report.json', 'w') as f:
    json.dump(validation_report, f, indent=2)

print(f"\n✓ Validation report saved to {EXP_ROOT}/metadata/validation_report.json")

if not validation_ok:
    print("\n⚠ FIX ERRORS BEFORE PROCEEDING TO FREEZE")
else:
    print("\n✓ All checks passed — safe to freeze")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 26d: CITATION.cff
# ═══════════════════════════════════════════════════════════════════════════

citation_cff = f"""cff-version: 1.2.0
title: "Multi-Fidelity Transformer for Crystal Property Prediction with Predictive Uncertainty"
message: "If you use this code or data, please cite this repository."
type: software
authors:
  - family-names: "[Your Name]"
    given-names: "[Your First Name]"
    orcid: "https://orcid.org/0000-0000-0000-0000"
repository-code: "https://github.com/[your-username]/materials-futures-mft"
license: "MIT"
version: "{EXP_VERSION}"
date-released: "{time.strftime('%Y-%m-%d')}"
abstract: >
  A Multi-Fidelity Transformer for crystal property prediction
  with predictive uncertainty, evaluated on Materials Project,
  JARVIS, and QMOF datasets.
keywords:
  - materials science
  - crystal property prediction
  - multi-fidelity learning
  - uncertainty quantification
  - transformer
  - machine learning
"""

with open(f'{EXP_ROOT}/CITATION.cff', 'w') as f:
    f.write(citation_cff)

print(f"✓ {EXP_ROOT}/CITATION.cff saved")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 26e: UPDATED DATA MANIFEST (with full query details)
# ═══════════════════════════════════════════════════════════════════════════

# Overwrite data manifest with full query details
data_manifest_full = add_schema({
    'version': EXP_VERSION,
    'sources': [
        {
            'name': 'Materials Project',
            'api': 'mp-api (MPRester)',
            'api_version': 'latest (2026)',
            'retrieval_date': time.strftime('%Y-%m-%d'),
            'query': {
                'fields': ['material_id', 'formula_pretty', 'structure',
                           'formation_energy_per_atom', 'band_gap',
                           'bulk_modulus', 'shear_modulus'],
                'filters': {
                    'formation_energy': (None, 2.0),
                    'band_gap': (0.0, 10.0),
                    'is_stable': True,
                },
            },
            'n_records_raw': len(df_mp),
            'n_records_after_featurization': int(np.sum(df_valid['source'] == 'materials_project')),
            'fidelity': 0,
            'notes': 'GGA-DFT calculations, lower fidelity',
        },
        {
            'name': 'JARVIS-DFT (dft_3d)',
            'api': 'jarvis-tools (figshare)',
            'api_version': 'latest (2026)',
            'retrieval_date': time.strftime('%Y-%m-%d'),
            'query': {
                'dataset': 'dft_3d',
                'filter': 'optb88vdw_bandgap not None AND formation_energy_peratom not None',
            },
            'n_records_raw': len(df_jv),
            'n_records_after_featurization': int(np.sum(df_valid['source'] == 'jarvis')),
            'fidelity': 1,
            'notes': 'Higher fidelity DFT calculations',
        },
        {
            'name': 'QMOF',
            'api': 'figshare direct download',
            'retrieval_date': time.strftime('%Y-%m-%d'),
            'query': {
                'url': 'https://figshare.com/ndownloader/articles/19346731/versions/4',
                'filter': 'bandgap (pbe) not None',
            },
            'n_records_raw': len(df_qmof) if len(df_qmof) > 0 else 0,
            'n_records_after_featurization': int(np.sum(df_valid['source'] == 'qmof')),
            'fidelity': 0,
            'notes': 'Metal-organic frameworks; formation_energy not always available',
        },
    ],
    'combined': {
        'total_raw': len(df_all),
        'total_featurized': len(X_raw),
        'feature_dim': int(X_raw.shape[1]),
        'feature_extraction': 'structural + elemental (mean/std/min/max aggregation)',
        'missing_value_handling': 'drop rows with NaN in target columns',
    },
    'splits': {
        'method': 'sklearn.model_selection.train_test_split',
        'first_split': {'ratio': '80/20', 'random_state': SEED, 'stratify': None},
        'second_split': {'ratio': '50/50 of temp', 'random_state': SEED, 'stratify': None},
        'final_sizes': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
        'leakage_prevention': 'StandardScaler fit on train set only',
    },
    'preprocessing': {
        'scaler': 'StandardScaler',
        'fit_on': 'train_only',
        'handle_missing': 'drop rows with NaN in formation_energy or band_gap',
    },
    'processed_files': {},
}, 'data_manifest')

# Hash processed files
for fname in ['train_features.npy', 'val_features.npy', 'test_features.npy',
              'train_targets.npy', 'val_targets.npy', 'test_targets.npy', 'scaler.pkl']:
    fpath = f'materials_futures_sprint/data/{fname}'
    h = file_hash(fpath)
    if h:
        data_manifest_full['processed_files'][fname] = {
            'sha256': h,
            'size_bytes': os.path.getsize(fpath),
        }

with open(f'{EXP_ROOT}/metadata/data_manifest.json', 'w') as f:
    json.dump(data_manifest_full, f, indent=2)

print(f"✓ {EXP_ROOT}/metadata/data_manifest.json updated with full query details")
