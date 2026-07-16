#!/usr/bin/env python3
"""
Reproduce script — loads checkpoints, regenerates metrics/tables/figures,
and verifies file hashes against the freeze manifest.

Usage:
    python reproduce.py [--exp-version v1]

What it does:
    1. Verifies all frozen file hashes match FREEZE_MANIFEST.json
    2. Loads saved predictions and recomputes metrics
    3. Regenerates tables (LaTeX + Markdown + CSV)
    4. Regenerates figures from figure_data CSVs
    5. Prints a summary report
"""

import json, os, sys, hashlib, argparse
import numpy as np
import pandas as pd

def file_hash(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(exp_root):
    """Verify all frozen file hashes match the manifest."""
    manifest_path = f'{exp_root}/FREEZE_MANIFEST.json'
    if not os.path.exists(manifest_path):
        print("  ⚠ No FREEZE_MANIFEST.json found — skipping hash verification")
        return True

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"=== Verifying file hashes ({manifest['version']}) ===")
    print(f"  Frozen at: {manifest['frozen_at']}")
    ok, fail, missing = 0, 0, 0
    for relpath, expected_hash in manifest['hashes'].items():
        fpath = f'{exp_root}/{relpath}'
        if not os.path.exists(fpath):
            print(f"  MISSING: {relpath}")
            missing += 1
            continue
        actual = file_hash(fpath)
        if actual == expected_hash:
            ok += 1
        else:
            print(f"  CHANGED: {relpath}")
            fail += 1

    print(f"  {ok} match, {fail} changed, {missing} missing")
    return fail == 0 and missing == 0


def regenerate_metrics(exp_root):
    """Load predictions and recompute metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("\n=== Regenerating metrics from predictions ===")
    pred_dir = f'{exp_root}/predictions'
    if not os.path.exists(pred_dir):
        print("  ⚠ No predictions/ directory found")
        return pd.DataFrame()

    pred_files = sorted([f for f in os.listdir(pred_dir)
                         if f.startswith('test_predictions_')])

    if not pred_files:
        print("  ⚠ No test prediction files found")
        return pd.DataFrame()

    results = []
    for pf in pred_files:
        df = pd.read_csv(f'{pred_dir}/{pf}')
        mae_fe = mean_absolute_error(df['true_fe'], df['pred_fe'])
        mae_bg = mean_absolute_error(df['true_bg'], df['pred_bg'])
        rmse_fe = np.sqrt(mean_squared_error(df['true_fe'], df['pred_fe']))
        rmse_bg = np.sqrt(mean_squared_error(df['true_bg'], df['pred_bg']))
        r2_fe = r2_score(df['true_fe'], df['pred_fe'])
        r2_bg = r2_score(df['true_bg'], df['pred_bg'])

        row = {
            'exp_id': df['exp_id'].iloc[0],
            'model': df['model'].iloc[0],
            'seed': int(df['seed'].iloc[0]),
            'MAE_FE': mae_fe, 'MAE_BG': mae_bg,
            'RMSE_FE': rmse_fe, 'RMSE_BG': rmse_bg,
            'R2_FE': r2_fe, 'R2_BG': r2_bg,
        }
        results.append(row)
        print(f"  {row['exp_id']}: MAE_FE={mae_fe:.4f}, MAE_BG={mae_bg:.4f}, R2_FE={r2_fe:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{exp_root}/tables/main_results_regenerated.csv', index=False)
    print(f"  Saved: {exp_root}/tables/main_results_regenerated.csv")
    return results_df


def regenerate_tables(exp_root, results_df):
    """Regenerate LaTeX and Markdown tables from results."""
    if results_df.empty:
        return

    print("\n=== Regenerating tables ===")

    # Group by model, compute mean ± std
    models = results_df.groupby('model')

    # LaTeX
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Test set performance (mean $\\pm$ std across seeds).}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\begin{tabular}{lcccc}\n\\hline\n"
    latex += "Model & MAE\\_FE (eV/atom) & MAE\\_BG (eV) & R$^2$\\_FE & R$^2$\\_BG \\\\\n\\hline\n"

    md = "| Model | MAE_FE (eV/atom) | MAE_BG (eV) | R²_FE | R²_BG |\n"
    md += "|-------|------------------|-------------|-------|-------|\n"

    for model_name, group in models:
        mae_fe = group['MAE_FE']
        mae_bg = group['MAE_BG']
        r2_fe = group['R2_FE']
        r2_bg = group['R2_BG']

        latex += (f"{model_name} & {mae_fe.mean():.4f}$\\pm${mae_fe.std():.4f} & "
                  f"{mae_bg.mean():.4f}$\\pm${mae_bg.std():.4f} & "
                  f"{r2_fe.mean():.4f}$\\pm${r2_fe.std():.4f} & "
                  f"{r2_bg.mean():.4f}$\\pm${r2_bg.std():.4f} \\\\\n")
        md += (f"| {model_name} | {mae_fe.mean():.4f}±{mae_fe.std():.4f} | "
               f"{mae_bg.mean():.4f}±{mae_bg.std():.4f} | "
               f"{r2_fe.mean():.4f}±{r2_fe.std():.4f} | "
               f"{r2_bg.mean():.4f}±{r2_bg.std():.4f} |\n")

    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"

    with open(f'{exp_root}/tables/results_table_regenerated.tex', 'w') as f:
        f.write(latex)
    with open(f'{exp_root}/tables/results_table_regenerated.md', 'w') as f:
        f.write(md)

    print(f"  Saved: results_table_regenerated.tex + .md")


def regenerate_figures(exp_root):
    """Regenerate figures from saved data CSVs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  ⚠ matplotlib not available — skipping figure regeneration")
        return

    fig_data_dir = f'{exp_root}/figure_data'
    fig_out_dir = f'{exp_root}/figures'

    if not os.path.exists(fig_data_dir):
        print("\n  ⚠ No figure_data/ directory found")
        return

    print("\n=== Regenerating figures from data ===")

    # Figure 1: Parity plots
    f1 = f'{fig_data_dir}/fig1_parity_data.csv'
    if os.path.exists(f1):
        df = pd.read_csv(f1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (col, label, color) in enumerate([
            ('fe', 'Formation Energy (eV/atom)', 'steelblue'),
            ('bg', 'Band Gap (eV)', 'darkorange')]):
            ax = axes[idx]
            ax.scatter(df[f'true_{col}'], df[f'pred_{col}'], alpha=0.4, s=15, c=color)
            lims = [min(df[f'true_{col}'].min(), df[f'pred_{col}'].min()),
                    max(df[f'true_{col}'].max(), df[f'pred_{col}'].max())]
            ax.plot(lims, lims, 'r--', linewidth=2, label='y=x')
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(df[f'true_{col}'], df[f'pred_{col}'])
            mae = mean_absolute_error(df[f'true_{col}'], df[f'pred_{col}'])
            ax.set_xlabel(f'True {label}')
            ax.set_ylabel(f'Predicted {label}')
            ax.set_title(f'({chr(97+idx)}) {label.split("(")[0].strip()}\nR²={r2:.3f}, MAE={mae:.4f}')
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'{fig_out_dir}/fig1_parity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ fig1_parity.png")

    # Figure 5: Spearman scatter
    f5 = f'{fig_data_dir}/fig5_spearman_data.csv'
    if os.path.exists(f5):
        df = pd.read_csv(f5)
        from scipy.stats import spearmanr
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (task, color) in enumerate([('fe', 'steelblue'), ('bg', 'darkorange')]):
            ax = axes[idx]
            unc = df[f'uncertainty_{task}']
            err = df[f'abs_error_{task}']
            rho, pval = spearmanr(unc, err)
            ax.scatter(unc, err, alpha=0.3, s=10, c=color)
            z = np.polyfit(unc, err, 1)
            p = np.poly1d(z)
            x_line = np.linspace(unc.min(), unc.max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2)
            ax.set_xlabel('Predicted Uncertainty (σ)')
            ax.set_ylabel('Absolute Error')
            task_label = 'Formation Energy' if task == 'fe' else 'Band Gap'
            ax.set_title(f'({chr(97+idx)}) {task_label}\nSpearman ρ={rho:.3f} (p={pval:.2e})')
        plt.tight_layout()
        plt.savefig(f'{fig_out_dir}/fig5_spearman.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ fig5_spearman.png")

    print("  (Other figures require re-running model inference)")


def print_summary(exp_root, results_df, hashes_ok):
    """Print final summary."""
    print(f"\n{'='*60}")
    print("REPRODUCTION SUMMARY")
    print(f"{'='*60}")
    print(f"Hash verification: {'✓ PASS' if hashes_ok else '✗ FAIL (files changed)'}")

    if not results_df.empty:
        print(f"\nMetrics from {len(results_df)} prediction files:")
        for model, group in results_df.groupby('model'):
            mae_fe = group['MAE_FE']
            mae_bg = group['MAE_BG']
            print(f"  {model}: MAE_FE={mae_fe.mean():.4f}±{mae_fe.std():.4f}, "
                  f"MAE_BG={mae_bg.mean():.4f}±{mae_bg.std():.4f}")

    print(f"\nNote: For full reproduction including model inference,")
    print(f"re-run the Kaggle notebook with the same seed and hyperparameters.")


def main():
    parser = argparse.ArgumentParser(description='Reproduce experiment results')
    parser.add_argument('--exp-version', default='v1', help='Experiment version (default: v1)')
    args = parser.parse_args()

    exp_root = f'experiments/{args.exp_version}'
    if not os.path.exists(exp_root):
        print(f"Error: {exp_root}/ not found")
        sys.exit(1)

    print(f"Reproducing experiment: {args.exp_version}\n")

    hashes_ok = verify_hashes(exp_root)
    results_df = regenerate_metrics(exp_root)
    regenerate_tables(exp_root, results_df)
    regenerate_figures(exp_root)
    print_summary(exp_root, results_df, hashes_ok)


if __name__ == '__main__':
    main()
