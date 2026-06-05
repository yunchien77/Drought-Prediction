"""Summarize ablation study results.

Reads result.json from each experiment directory and produces:
  - ablation/results/summary.csv   — full table with OOF MAE per experiment
  - ablation/results/summary.md    — markdown table for easy reading
  - console printout grouped by experiment group

Usage:
    python summarize_results.py
"""
import json
import math
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Group labels
# ─────────────────────────────────────────────────────────────────────────────
GROUP_LABELS = {
    "A": "Module Ablations",
    "B": "Hyperparameter Ablations",
    "C": "Window Size Ablations",
    "D": "Architecture & Weight Ablations",
}

EXPERIMENT_GROUPS = {
    "baseline":                   "A",
    "no_preprocessing":           "A",
    "no_gap_stratified":          "A",
    "no_proxy_scores":            "A",
    "no_score_lag":               "A",
    "no_physical_indices":        "A",
    "no_anomaly_features":        "A",
    "no_region_stats":            "A",
    "no_seasonal_encoding":       "A",
    "no_test_season_features":    "A",
    "no_calendar_matched_val":    "A",
    "no_kaggle_proxy_val":        "A",
    "no_sample_weights":          "A",
    "no_calibration":             "A",
    "with_calendar_season_weights": "A",
    "lr_0.01":                    "B",
    "lr_0.03":                    "B",
    "leaves_127":                 "B",
    "leaves_511":                 "B",
    "feature_fraction_0.5":       "B",
    "feature_fraction_0.9":       "B",
    "lambda_high":                "B",
    "lambda_low":                 "B",
    "early_stop_50":              "B",
    "early_stop_300":             "B",
    "min_child_10":               "B",
    "min_child_50":               "B",
    "bagging_0.6":                "B",
    "windows_short":              "C",
    "windows_no_long":            "C",
    "windows_only_long":          "C",
    "folds_2":                    "D",
    "folds_6":                    "D",
    "gap_threshold_8":            "D",
    "gap_threshold_20":           "D",
    "weight_nonzero_2.0":         "D",
    "weight_severe_5.0":          "D",
}


def _fmt(v, digits=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{digits}f}"


def load_results() -> list[dict]:
    results = []
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        rj = exp_dir / "result.json"
        if rj.is_file():
            with open(rj) as f:
                results.append(json.load(f))
    return results


def build_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        name    = r.get("name", "?")
        fw_m    = r.get("fw_maes", {})
        rows.append({
            "name":        name,
            "group":       EXPERIMENT_GROUPS.get(name, "?"),
            "group_label": GROUP_LABELS.get(EXPERIMENT_GROUPS.get(name, "?"), "Other"),
            "status":      r.get("status", "?"),
            "oof_mae":     r.get("oof_mae"),
            "fw1_mae":     float(fw_m.get("1", float("nan"))) if "1" in fw_m else None,
            "fw2_mae":     float(fw_m.get("2", float("nan"))) if "2" in fw_m else None,
            "fw3_mae":     float(fw_m.get("3", float("nan"))) if "3" in fw_m else None,
            "fw4_mae":     float(fw_m.get("4", float("nan"))) if "4" in fw_m else None,
            "fw5_mae":     float(fw_m.get("5", float("nan"))) if "5" in fw_m else None,
            "elapsed_min": r.get("elapsed_min"),
            "description": r.get("description", ""),
            "error":       r.get("error", ""),
        })
    df = pd.DataFrame(rows)

    # Delta vs baseline
    bl = df[df["name"] == "baseline"]
    if len(bl) > 0 and bl.iloc[0]["oof_mae"] is not None:
        base_mae = float(bl.iloc[0]["oof_mae"])
        df["delta_vs_baseline"] = df["oof_mae"].apply(
            lambda x: round(x - base_mae, 4) if x is not None else None
        )
    else:
        df["delta_vs_baseline"] = None

    return df


def print_group_table(df: pd.DataFrame, group: str):
    label = GROUP_LABELS.get(group, group)
    gdf   = df[df["group"] == group].copy()
    # baseline first, then by oof_mae
    gdf["_s"] = gdf["name"].apply(lambda n: 0 if n == "baseline" else 1)
    gdf = gdf.sort_values(["_s", "oof_mae"]).drop(columns=["_s"])

    print(f"\n{'─'*80}")
    print(f"  Group {group}: {label}  ({len(gdf)} experiments)")
    print(f"{'─'*80}")
    header = f"  {'Name':<35} {'OOF MAE':>8}  {'Δ baseline':>10}  {'fw1':>6} {'fw2':>6} {'fw3':>6} {'fw4':>6} {'fw5':>6}  {'Status'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in gdf.iterrows():
        delta_str = f"{row['delta_vs_baseline']:+.4f}" if row["delta_vs_baseline"] is not None else "  N/A  "
        print(
            f"  {row['name']:<35} {_fmt(row['oof_mae']):>8}  {delta_str:>10}  "
            f"{_fmt(row['fw1_mae']):>6} {_fmt(row['fw2_mae']):>6} "
            f"{_fmt(row['fw3_mae']):>6} {_fmt(row['fw4_mae']):>6} "
            f"{_fmt(row['fw5_mae']):>6}  {row['status']}"
        )


def write_markdown(df: pd.DataFrame, path: Path):
    lines = ["# Ablation Study Results\n"]
    df_bl = df[df["name"] == "baseline"]
    if len(df_bl) > 0 and df_bl.iloc[0]["oof_mae"] is not None:
        lines.append(f"**Baseline OOF MAE**: {_fmt(df_bl.iloc[0]['oof_mae'])}\n")

    for group in sorted(df["group"].unique()):
        label = GROUP_LABELS.get(group, group)
        lines.append(f"\n## Group {group}: {label}\n")
        lines.append("| Experiment | OOF MAE | Δ Baseline | fw1 | fw2 | fw3 | fw4 | fw5 | Status |")
        lines.append("|:-----------|--------:|-----------:|----:|----:|----:|----:|----:|:-------|")

        gdf = df[df["group"] == group].copy()
        gdf["_s"] = gdf["name"].apply(lambda n: 0 if n == "baseline" else 1)
        gdf = gdf.sort_values(["_s", "oof_mae"]).drop(columns=["_s"])

        for _, row in gdf.iterrows():
            delta = f"{row['delta_vs_baseline']:+.4f}" if row["delta_vs_baseline"] is not None else "N/A"
            lines.append(
                f"| `{row['name']}` | {_fmt(row['oof_mae'])} | {delta} | "
                f"{_fmt(row['fw1_mae'])} | {_fmt(row['fw2_mae'])} | "
                f"{_fmt(row['fw3_mae'])} | {_fmt(row['fw4_mae'])} | "
                f"{_fmt(row['fw5_mae'])} | {row['status']} |"
            )

    lines.append("\n---\n")
    lines.append("*Generated by ablation/summarize_results.py*\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report → {path}")


def main():
    results = load_results()
    if not results:
        print(f"No result.json files found in {RESULTS_DIR}")
        return

    print(f"Found {len(results)} experiment results")

    df = build_dataframe(results)

    # Save CSV
    csv_path = RESULTS_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV → {csv_path}")

    # Print grouped tables
    print("\n" + "=" * 80)
    print("ABLATION STUDY — LOCAL VALIDATION (OOF MAE) RESULTS")
    print("=" * 80)
    for group in sorted(df["group"].unique()):
        print_group_table(df, group)

    # Errors summary
    errors = df[df["status"] == "error"]
    if len(errors) > 0:
        print(f"\n⚠  {len(errors)} experiments failed:")
        for _, row in errors.iterrows():
            print(f"   {row['name']}: {row['error'][:80]}")

    # Best & worst
    done = df[df["status"] == "done"].dropna(subset=["oof_mae"])
    if len(done) > 0:
        best  = done.loc[done["oof_mae"].idxmin()]
        worst = done.loc[done["oof_mae"].idxmax()]
        print(f"\n  Best  OOF MAE: {best['name']:<35} {_fmt(best['oof_mae'])}")
        print(f"  Worst OOF MAE: {worst['name']:<35} {_fmt(worst['oof_mae'])}")

    # Write markdown
    write_markdown(df, RESULTS_DIR / "summary.md")


if __name__ == "__main__":
    main()
