"""Ablation study orchestrator.

Runs all experiments sequentially, each as an isolated subprocess.
Results are saved to ablation/results/.

Usage:
    python ablation_study.py                    # run all experiments
    python ablation_study.py --only baseline    # run a single experiment
    python ablation_study.py --skip baseline    # skip specific experiments
    python ablation_study.py --list             # list experiments and exit
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ABLATION_DIR = Path(__file__).parent
RESULTS_DIR  = ABLATION_DIR / "results"
RUNNER       = ABLATION_DIR / "run_experiment.py"


# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────────────────────────────────────

def define_experiments() -> list[dict]:
    """
    Each experiment is fully isolated: separate models dir + submission CSV.

    Fields
    ------
    name              : unique identifier (used for filenames)
    group             : one of A/B/C/D (module / hyperparam / window / architecture)
    description       : human-readable description
    config_overrides  : dict of config attribute overrides
    lgbm_overrides    : dict of LightGBM param overrides
    zero_feature_groups : list of substring patterns — matching feature columns
                          are zeroed in X (train + test) AFTER feature building.
                          Safe with multiprocessing because zeroing runs in main process.
    """
    return [
        # ══════════════════════════════════════════════════════════════════════
        # Group A – Module ablations
        # ══════════════════════════════════════════════════════════════════════
        {
            "name":        "baseline",
            "group":       "A",
            "description": "Baseline: all modules on, default config",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "no_preprocessing",
            "group":       "A",
            "description": "Ablation: disable log1p/sqrt preprocessing on prec/surf_pre",
            "config_overrides":    {"USE_PREPROCESSING": False},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "no_gap_stratified",
            "group":       "A",
            "description": "Ablation: disable short/long-gap stratified sub-models",
            "config_overrides":    {"USE_GAP_STRATIFIED": False},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "no_proxy_scores",
            "group":       "A",
            "description": "Ablation: zero proxy score features (proxy_p7/21/91/main/diffs)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": ["proxy_"],
        },
        {
            "name":        "no_score_lag",
            "group":       "A",
            "description": "Ablation: zero score-lag features (last_known_score, gap_weeks, score_lag_decayed)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": ["last_known_score", "gap_weeks", "score_lag_decayed"],
        },
        {
            "name":        "no_physical_indices",
            "group":       "A",
            "description": "Ablation: zero physical drought index features (VPD, DTR, heat-DD, aridity, etc.)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": [
                "dry_day_frac", "heat_dd", "vpd_mean", "vpd_max",
                "hum_aridity_idx", "prec_deficit", "dtr_mean", "prec_sum",
            ],
        },
        {
            "name":        "no_anomaly_features",
            "group":       "A",
            "description": "Ablation: zero anomaly z-score features (*_anom_*, *_mo_anom_*)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": ["_anom_"],
        },
        {
            "name":        "no_region_stats",
            "group":       "A",
            "description": "Ablation: zero region-level score statistics features",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": [
                "reg_mean", "reg_std", "reg_q25", "reg_q75",
                "reg_q90", "reg_nonzero", "reg_score_mo_mean", "fw_score_mo_mean",
            ],
        },
        {
            "name":        "no_seasonal_encoding",
            "group":       "A",
            "description": "Ablation: zero seasonal encoding features (sin/cos DOY/month, quarter)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": [
                "sin_doy", "cos_doy", "sin_month", "cos_month",
                "month_raw", "quarter",
            ],
        },
        {
            "name":        "no_test_season_features",
            "group":       "A",
            "description": "Ablation: zero test-season distance features (month_dist_to_test, is_test_season)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": ["month_dist_to_test", "is_test_season"],
        },
        {
            "name":        "no_calendar_matched_val",
            "group":       "A",
            "description": "Ablation: disable calendar-matched early-stopping validation",
            "config_overrides":    {
                "USE_CALENDAR_MATCHED_VAL": False,
                "USE_KAGGLE_PROXY_VAL":     False,
            },
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "no_kaggle_proxy_val",
            "group":       "A",
            "description": "Ablation: disable Kaggle proxy holdout (keep calendar-matched val only)",
            "config_overrides":    {"USE_KAGGLE_PROXY_VAL": False},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "no_sample_weights",
            "group":       "A",
            "description": "Ablation: uniform sample weights (disable severity upweighting)",
            "config_overrides":    {"USE_SAMPLE_WEIGHT": False},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "no_calibration",
            "group":       "A",
            "description": "Ablation: disable isotonic regression post-calibration",
            "config_overrides":    {"USE_CALIBRATION": False},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "with_calendar_season_weights",
            "group":       "A",
            "description": "Test: enable in-season training-weight boost (disabled in baseline)",
            "config_overrides":    {"USE_CALENDAR_SEASON_WEIGHTS": True},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        # ══════════════════════════════════════════════════════════════════════
        # Group B – Hyperparameter ablations
        # ══════════════════════════════════════════════════════════════════════
        {
            "name":        "lr_0.01",
            "group":       "B",
            "description": "Hyperparam: learning_rate=0.01 (slower, smoother)",
            "config_overrides":    {},
            "lgbm_overrides":      {"learning_rate": 0.01},
            "zero_feature_groups": [],
        },
        {
            "name":        "lr_0.03",
            "group":       "B",
            "description": "Hyperparam: learning_rate=0.03 (faster, coarser)",
            "config_overrides":    {},
            "lgbm_overrides":      {"learning_rate": 0.03},
            "zero_feature_groups": [],
        },
        {
            "name":        "leaves_127",
            "group":       "B",
            "description": "Hyperparam: num_leaves=127 (shallower trees, less overfit)",
            "config_overrides":    {},
            "lgbm_overrides":      {"num_leaves": 127},
            "zero_feature_groups": [],
        },
        {
            "name":        "leaves_511",
            "group":       "B",
            "description": "Hyperparam: num_leaves=511 (deeper trees, more capacity)",
            "config_overrides":    {},
            "lgbm_overrides":      {"num_leaves": 511},
            "zero_feature_groups": [],
        },
        {
            "name":        "feature_fraction_0.5",
            "group":       "B",
            "description": "Hyperparam: feature_fraction=0.5 (more regularization via col subsampling)",
            "config_overrides":    {},
            "lgbm_overrides":      {"feature_fraction": 0.5},
            "zero_feature_groups": [],
        },
        {
            "name":        "feature_fraction_0.9",
            "group":       "B",
            "description": "Hyperparam: feature_fraction=0.9 (less col subsampling)",
            "config_overrides":    {},
            "lgbm_overrides":      {"feature_fraction": 0.9},
            "zero_feature_groups": [],
        },
        {
            "name":        "lambda_high",
            "group":       "B",
            "description": "Hyperparam: lambda_l1=0.3, lambda_l2=0.3 (stronger regularization)",
            "config_overrides":    {},
            "lgbm_overrides":      {"lambda_l1": 0.3, "lambda_l2": 0.3},
            "zero_feature_groups": [],
        },
        {
            "name":        "lambda_low",
            "group":       "B",
            "description": "Hyperparam: lambda_l1=0.01, lambda_l2=0.01 (weaker regularization)",
            "config_overrides":    {},
            "lgbm_overrides":      {"lambda_l1": 0.01, "lambda_l2": 0.01},
            "zero_feature_groups": [],
        },
        {
            "name":        "early_stop_50",
            "group":       "B",
            "description": "Hyperparam: early_stopping_rounds=50 (aggressive early stop)",
            "config_overrides":    {"EARLY_STOPPING_ROUNDS": 50},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "early_stop_300",
            "group":       "B",
            "description": "Hyperparam: early_stopping_rounds=300 (patient early stop)",
            "config_overrides":    {"EARLY_STOPPING_ROUNDS": 300},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "min_child_10",
            "group":       "B",
            "description": "Hyperparam: min_child_samples=10 (smaller leaves allowed)",
            "config_overrides":    {},
            "lgbm_overrides":      {"min_child_samples": 10},
            "zero_feature_groups": [],
        },
        {
            "name":        "min_child_50",
            "group":       "B",
            "description": "Hyperparam: min_child_samples=50 (larger leaves required)",
            "config_overrides":    {},
            "lgbm_overrides":      {"min_child_samples": 50},
            "zero_feature_groups": [],
        },
        {
            "name":        "bagging_0.6",
            "group":       "B",
            "description": "Hyperparam: bagging_fraction=0.6 (more row subsampling)",
            "config_overrides":    {},
            "lgbm_overrides":      {"bagging_fraction": 0.6},
            "zero_feature_groups": [],
        },
        # ══════════════════════════════════════════════════════════════════════
        # Group C – Window size ablations
        # Implemented via feature-column zeroing (safe with multiprocessing):
        # windows with w=42 are zeroed for "short" variant; w=91 zeroed for "no_long".
        # The baseline window set is [7, 14, 21, 42, 91].
        # ══════════════════════════════════════════════════════════════════════
        {
            "name":        "windows_short",
            "group":       "C",
            "description": "Window ablation: remove 42d and 91d window features (keep 7/14/21 only)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": ["_42d", "_91d"],
        },
        {
            "name":        "windows_no_long",
            "group":       "C",
            "description": "Window ablation: remove 91d window features (keep 7/14/21/42)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": ["_91d"],
        },
        {
            "name":        "windows_only_long",
            "group":       "C",
            "description": "Window ablation: remove 7d and 14d window features (keep 21/42/91)",
            "config_overrides":    {},
            "lgbm_overrides":      {},
            "zero_feature_groups": ["_7d", "_14d"],
        },
        # ══════════════════════════════════════════════════════════════════════
        # Group D – Training architecture
        # ══════════════════════════════════════════════════════════════════════
        {
            "name":        "folds_2",
            "group":       "D",
            "description": "Architecture: N_PH_FOLDS=2 (fewer walk-forward folds)",
            "config_overrides":    {"N_PH_FOLDS": 2},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "folds_6",
            "group":       "D",
            "description": "Architecture: N_PH_FOLDS=6 (more walk-forward folds)",
            "config_overrides":    {"N_PH_FOLDS": 6},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "gap_threshold_8",
            "group":       "D",
            "description": "Architecture: GAP_SHORT_THRESHOLD=8 weeks (tighter short-gap boundary)",
            "config_overrides":    {"GAP_SHORT_THRESHOLD": 8},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "gap_threshold_20",
            "group":       "D",
            "description": "Architecture: GAP_SHORT_THRESHOLD=20 weeks (wider short-gap boundary)",
            "config_overrides":    {"GAP_SHORT_THRESHOLD": 20},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "weight_nonzero_2.0",
            "group":       "D",
            "description": "Architecture: WEIGHT_NONZERO=2.0 (higher upweight for non-zero drought)",
            "config_overrides":    {"WEIGHT_NONZERO": 2.0},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
        {
            "name":        "weight_severe_5.0",
            "group":       "D",
            "description": "Architecture: WEIGHT_SEVERE=5.0 (higher upweight for severe drought ≥3)",
            "config_overrides":    {"WEIGHT_SEVERE": 5.0},
            "lgbm_overrides":      {},
            "zero_feature_groups": [],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp: dict) -> dict:
    """Launch one experiment as a subprocess. Returns result dict."""
    name      = exp["name"]
    exp_dir   = RESULTS_DIR / name
    models_dir = exp_dir / "models"
    result_json = exp_dir / "result.json"
    output_csv  = RESULTS_DIR / f"submission_{name}.csv"

    exp_payload = dict(exp)
    exp_payload["models_dir"]   = str(models_dir)
    exp_payload["output_csv"]   = str(output_csv)
    exp_payload["result_json"]  = str(result_json)

    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*70}")
    print(f"Running: {name}  [{exp.get('group','')}] — {exp.get('description','')}")
    print(f"{'─'*70}")

    t0  = time.time()
    cmd = [sys.executable, str(RUNNER), json.dumps(exp_payload)]
    proc = subprocess.run(
        cmd,
        capture_output=False,   # let stdout/stderr flow to terminal
        text=True,
    )
    elapsed = (time.time() - t0) / 60

    if result_json.exists():
        with open(result_json) as f:
            result = json.load(f)
    else:
        result = {
            "name":       name,
            "status":     "error",
            "error":      f"result.json not written (exit code {proc.returncode})",
            "oof_mae":    float("nan"),
            "elapsed_min": round(elapsed, 2),
        }
        with open(result_json, "w") as f:
            json.dump(result, f, indent=2)

    print(f"  → {name}: OOF MAE={result.get('oof_mae','N/A')}  "
          f"status={result.get('status')}  {elapsed:.1f} min")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary compilation
# ─────────────────────────────────────────────────────────────────────────────

def compile_summary(results: list[dict], experiments: list[dict]):
    """Write summary CSV + print table."""
    group_map = {e["name"]: e.get("group", "?") for e in experiments}

    rows = []
    for r in results:
        name     = r["name"]
        oof_mae  = r.get("oof_mae", float("nan"))
        fw_maes  = r.get("fw_maes", {})
        status   = r.get("status", "?")
        elapsed  = r.get("elapsed_min", float("nan"))
        group    = group_map.get(name, "?")
        desc     = r.get("description", "")

        row = {
            "name":        name,
            "group":       group,
            "status":      status,
            "oof_mae":     round(oof_mae, 4) if oof_mae == oof_mae else None,
            "fw1_mae":     round(float(fw_maes.get("1", float("nan"))), 4)
                           if "1" in fw_maes else None,
            "fw2_mae":     round(float(fw_maes.get("2", float("nan"))), 4)
                           if "2" in fw_maes else None,
            "fw3_mae":     round(float(fw_maes.get("3", float("nan"))), 4)
                           if "3" in fw_maes else None,
            "fw4_mae":     round(float(fw_maes.get("4", float("nan"))), 4)
                           if "4" in fw_maes else None,
            "fw5_mae":     round(float(fw_maes.get("5", float("nan"))), 4)
                           if "5" in fw_maes else None,
            "elapsed_min": round(elapsed, 1),
            "description": desc,
        }
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)

    # Compute delta vs baseline
    baseline_row = df[df["name"] == "baseline"]
    if len(baseline_row) > 0:
        base_mae = float(baseline_row.iloc[0]["oof_mae"])
        df["delta_vs_baseline"] = (df["oof_mae"] - base_mae).round(4)
    else:
        df["delta_vs_baseline"] = None

    # Sort: baseline first, then by OOF MAE
    df["_sort_key"] = df["name"].apply(lambda n: 0 if n == "baseline" else 1)
    df = df.sort_values(["_sort_key", "oof_mae"]).drop(columns=["_sort_key"])

    out_csv = RESULTS_DIR / "summary.csv"
    df.to_csv(out_csv, index=False)

    # Print table
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    cols = ["name", "group", "oof_mae", "delta_vs_baseline", "status", "elapsed_min"]
    print(df[cols].to_string(index=False))
    print("=" * 80)
    print(f"\nSummary saved → {out_csv}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Ablation study orchestrator")
    p.add_argument("--only",  nargs="+", metavar="NAME",
                   help="Run only these experiment names")
    p.add_argument("--skip",  nargs="+", metavar="NAME",
                   help="Skip these experiment names")
    p.add_argument("--group", nargs="+", metavar="GROUP",
                   help="Run only experiments in these groups (A/B/C/D)")
    p.add_argument("--list",  action="store_true",
                   help="List experiments and exit")
    return p.parse_args()


def main():
    args        = _parse_args()
    experiments = define_experiments()

    if args.list:
        print(f"{'Name':<35} {'Grp':<4} Description")
        print("-" * 90)
        for e in experiments:
            print(f"{e['name']:<35} {e.get('group','?'):<4} {e.get('description','')}")
        return

    # Filter
    selected = experiments
    if args.only:
        selected = [e for e in selected if e["name"] in args.only]
    if args.skip:
        selected = [e for e in selected if e["name"] not in args.skip]
    if args.group:
        selected = [e for e in selected if e.get("group") in args.group]

    print(f"\nAblation Study — {len(selected)} experiments")
    print(f"Results → {RESULTS_DIR}\n")

    t_total = time.time()
    results = []
    for i, exp in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}]", end="")
        r = run_experiment(exp)
        results.append(r)

    # Compile summary using ALL result JSONs (including previously run experiments)
    all_results = []
    for exp in experiments:
        rj = RESULTS_DIR / exp["name"] / "result.json"
        if rj.exists():
            with open(rj) as f:
                all_results.append(json.load(f))

    compile_summary(all_results, experiments)

    total_elapsed = (time.time() - t_total) / 60
    print(f"\nAll done in {total_elapsed:.1f} min")


if __name__ == "__main__":
    main()
