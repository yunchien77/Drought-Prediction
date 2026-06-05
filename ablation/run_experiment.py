"""Single ablation experiment runner.

Called as a subprocess by ablation_study.py. Receives a JSON experiment
spec via argv[1]. Patches config *before* importing any pipeline module so
that even module-level constants (WINDOW_SIZES, N_WORKERS, USE_* flags) see
the ablation values in every code path, including multiprocessing workers
(which we redirect to run in-process via _InProcessPool).
"""
import sys
import json
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# In-process pool – ensures config patches are visible to all feature workers
# ─────────────────────────────────────────────────────────────────────────────

import multiprocessing as _mp


class _InProcessPool:
    """Drop-in replacement for mp.Pool that runs workers in the current process.

    This guarantees that any config attribute we patched before the import is
    seen by make_features / _worker_train_region / _worker_test_region without
    spawning a fresh interpreter that would reload the original config.
    """

    def __init__(self, processes=None, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def imap(self, fn, iterable, chunksize=1):
        return iter([fn(x) for x in iterable])

    def imap_unordered(self, fn, iterable, chunksize=1):
        return iter([fn(x) for x in iterable])

    def map(self, fn, iterable, chunksize=None):
        return [fn(x) for x in iterable]

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


# Patch mp.Pool BEFORE any other import so data_pipeline sees _InProcessPool.
_mp.Pool = _InProcessPool


# ─────────────────────────────────────────────────────────────────────────────
# Config patching (must happen before importing pipeline modules)
# ─────────────────────────────────────────────────────────────────────────────

def _setup(exp: dict):
    code_dir = Path(__file__).parent.parent / "code"
    sys.path.insert(0, str(code_dir))

    import config

    models_dir = Path(exp["models_dir"])
    eval_dir   = models_dir.parent / "eval"
    for d in [
        models_dir,
        models_dir / "per_horizon",
        models_dir / "gap_short",
        models_dir / "gap_long",
        eval_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Isolated artifact paths
    config.MODELS_DIR             = models_dir
    config.MODELS_PH_DIR          = models_dir / "per_horizon"
    config.MODELS_GAP_SHORT_DIR   = models_dir / "gap_short"
    config.MODELS_GAP_LONG_DIR    = models_dir / "gap_long"
    config.CLIMATOLOGY_PATH       = models_dir / "climatology.pkl"
    config.PROXY_RIDGE_PATH       = models_dir / "proxy_ridge.pkl"
    config.MODELS_PKL_PATH        = models_dir / "lgbm_models.pkl"
    config.CALIBRATOR_PATH        = models_dir / "calibrator.pkl"
    config.FEATURE_NAMES_PATH     = models_dir / "feature_names.json"
    config.EVAL_REPORT_PATH       = eval_dir   / "eval_report.json"
    config.PREPROC_PATH           = models_dir / "preprocessing.pkl"

    # Ablation-safe defaults
    config.USE_CACHE              = False   # never use shared feature cache
    config.PARALLEL_FW_TRAINING   = False   # sequential: safe with patched config
    config.N_WORKERS              = 1       # single worker: uses _InProcessPool

    # Experiment-specific overrides
    for k, v in exp.get("config_overrides", {}).items():
        setattr(config, k, v)

    # LightGBM hyperparameter overrides
    if exp.get("lgbm_overrides"):
        new_params = dict(config.LGBM_PARAMS)
        new_params.update(exp["lgbm_overrides"])
        config.LGBM_PARAMS = new_params

    # Recompute derived fields after patching.
    # IMPORTANT: Use the same n_jobs formula as the main pipeline (parallel mode):
    #   LGBM_JOBS = CPU_WORKERS // N_PRED
    # The main pipeline runs 5 fw-models in parallel, each getting CPU_WORKERS//N_PRED
    # threads.  Even though ablation is sequential, we must pass the *same* n_jobs
    # so that LightGBM's internal random state (bagging / feature-fraction sampling)
    # is identical — different thread counts produce different trees even with the
    # same seed, which would inflate the baseline MAE vs. the real pipeline.
    from hardware import CPU_WORKERS
    config.LGBM_JOBS              = max(1, CPU_WORKERS // config.N_PRED)
    config.LGBM_PARAMS["n_jobs"]  = config.LGBM_JOBS
    config.LGBM_PARAMS["seed"]    = config.SEED

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_region_gap_dict(config) -> dict[str, float]:
    """Reconstruct region-gap mapping from train/test dates."""
    test  = pd.read_csv(config.TEST_PATH,  usecols=["region_id", "date"])
    train = pd.read_csv(config.TRAIN_PATH, usecols=["region_id", "date"])
    train_ends  = train.groupby("region_id")["date"].last()
    test_starts = test.groupby("region_id")["date"].first()

    def _parse(s):
        parts = str(s).strip().split("T")[0].split(" ")[0].split("-")
        return int(parts[0]), int(parts[1]), int(parts[2])

    out = {}
    for r in test["region_id"].unique():
        if r not in train_ends.index:
            out[str(r)] = 52.0
            continue
        try:
            te_y, te_m, te_d = _parse(train_ends[r])
            ts_y, ts_m, ts_d = _parse(test_starts[r])
            gw = ((ts_y - te_y) * 365 + (ts_m - te_m) * 30 + (ts_d - te_d)) / 7.0
            out[str(r)] = float(gw)
        except Exception:
            out[str(r)] = 52.0
    return out


def _zero_features(arr: np.ndarray, patterns: list[str], feat_names: list[str]) -> np.ndarray:
    """Zero columns whose name contains any pattern string."""
    cols = [i for i, n in enumerate(feat_names)
            if any(p in n for p in patterns)]
    if cols:
        arr = arr.copy()
        arr[:, cols] = 0.0
        print(f"  Zeroed {len(cols)} feature columns matching {patterns}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    exp         = json.loads(sys.argv[1])
    name        = exp["name"]
    zero_groups = exp.get("zero_feature_groups", [])
    output_csv  = Path(exp["output_csv"])
    result_json = Path(exp["result_json"])

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {name}")
    print(f"  {exp.get('description', '')}")
    print(f"{'='*70}\n")

    config = _setup(exp)

    t0     = time.time()
    result = {"name": name, "description": exp.get("description", ""),
              "status": "running"}

    try:
        from train import (
            load_data, validate_regions_and_gap,
            stage0_preprocessing, stage1_climatology,
            stage2_proxy_ridge, stage3_build_features, stage4_train,
        )
        from model import fit_calibrator, save_models, predict_for_test
        from features import build_feature_names
        from data_pipeline import build_test_features
        from sklearn.metrics import mean_absolute_error

        # ── Stage 0: Preprocessing ────────────────────────────────────────────
        preproc = stage0_preprocessing(force=True) if config.USE_PREPROCESSING else None

        # ── Load train ────────────────────────────────────────────────────────
        train = load_data()
        _, region_gap_dict, region_test_months = validate_regions_and_gap(train)

        # ── Stage 1: Climatology ──────────────────────────────────────────────
        clim       = stage1_climatology(train, force=True)
        clim_dict  = clim["clim_dict"]
        month_dict = clim["month_dict"]
        sstat_dict = clim["sstat_dict"]
        smo_dict   = clim["smo_dict"]

        # ── Stage 2: Proxy Ridge ──────────────────────────────────────────────
        proxy_ridge = stage2_proxy_ridge(train, clim_dict, force=True)

        # ── Stage 3: Feature matrix ───────────────────────────────────────────
        stage_result = stage3_build_features(
            train, clim_dict, month_dict, sstat_dict, smo_dict, proxy_ridge,
            preproc_artifacts=preproc,
            region_gap_dict=region_gap_dict,
            region_test_months=region_test_months,
            force=True,
        )
        X, y, groups, time_keys, fw_keys, month_keys = stage_result
        feat_names  = build_feature_names()

        # ── Feature ablation: zero specified columns ──────────────────────────
        zeroed_cols = []
        if zero_groups:
            zeroed_cols = [i for i, n in enumerate(feat_names)
                           if any(p in n for p in zero_groups)]
            if zeroed_cols:
                X = X.copy()
                X[:, zeroed_cols] = 0.0
                print(f"  Zeroed {len(zeroed_cols)} train feature columns")

        # ── Stage 4: Training ─────────────────────────────────────────────────
        models, eval_report, oof_preds, _ = stage4_train(
            X, y, groups, time_keys, fw_keys,
            month_keys=month_keys,
            adversarial_weights=None,
            region_test_months=region_test_months,
            region_gap_dict=region_gap_dict,
        )

        valid   = ~np.isnan(oof_preds)
        oof_mae = (
            float(mean_absolute_error(y[valid], np.clip(oof_preds[valid], 0, 5)))
            if valid.sum() > 0 else float("nan")
        )
        print(f"\n  OOF MAE: {oof_mae:.4f}")

        fw_maes = eval_report.get("fw_maes", {})
        result["oof_mae"]  = oof_mae
        result["fw_maes"]  = {str(k): float(v) for k, v in fw_maes.items()}

        # ── Calibration + save ────────────────────────────────────────────────
        calibrator = fit_calibrator(oof_preds, y)
        save_models(models, calibrator, oof_mae)
        del train; gc.collect()

        # ── Test features ─────────────────────────────────────────────────────
        test = pd.read_csv(
            config.TEST_PATH,
            dtype={c: "float32" for c in config.METEO_COLS},
        )
        test = test.sort_values(["region_id", "date"]).reset_index(drop=True)
        test["month"] = test["date"].str.split("-").str[1].astype("int8")
        region_test_months_test = {
            str(r): int(m)
            for r, m in test.groupby("region_id")["month"].first().items()
        }

        test_features = build_test_features(
            test, clim_dict, month_dict, sstat_dict, smo_dict, proxy_ridge,
            preproc_artifacts=preproc,
            region_test_months=region_test_months_test,
        )

        # Zero same columns in test features
        if zeroed_cols:
            for region_key, feat_arr in test_features.items():
                feat_arr[:, zeroed_cols] = 0.0

        # ── Prediction ────────────────────────────────────────────────────────
        fallback = float(
            pd.read_csv(config.TRAIN_PATH, usecols=["score"])
            ["score"].dropna().mean()
        )

        final_preds: dict = {}
        if config.USE_GAP_STRATIFIED:
            try:
                from gap_stratified import (
                    load_stratified_models, predict_stratified,
                    compute_region_strata,
                )
                rgd = _build_region_gap_dict(config)
                short_m, long_m = load_stratified_models()
                if short_m and long_m:
                    strata      = compute_region_strata(rgd)
                    final_preds = predict_stratified(
                        test_features, short_m, long_m, strata, rgd,
                        fallback=fallback,
                    )
                    print("  Prediction: gap-stratified")
            except Exception as e:
                print(f"  Gap-stratified prediction failed ({e}), using main model")

        if not final_preds:
            final_preds = predict_for_test(
                test_features, models, calibrator, fallback=fallback,
            )
            print("  Prediction: main LightGBM")

        # ── Write submission ──────────────────────────────────────────────────
        sample    = pd.read_csv(config.SAMPLE_PATH)
        pred_cols = [f"pred_week{i}" for i in range(1, config.N_PRED + 1)]
        rows = []
        for _, row in sample.iterrows():
            rid = row["region_id"]
            p   = final_preds.get(rid, [fallback] * config.N_PRED)
            rows.append({
                "region_id":  rid,
                "pred_week1": round(float(p[0]), 4),
                "pred_week2": round(float(p[1]), 4),
                "pred_week3": round(float(p[2]), 4),
                "pred_week4": round(float(p[3]), 4),
                "pred_week5": round(float(p[4]), 4),
            })
        sub = pd.DataFrame(rows)
        assert (sub[pred_cols] >= 0).all().all(), "Negative predictions"
        assert (sub[pred_cols] <= 5).all().all(),  "Predictions > 5"
        assert len(sub) == len(sample), "Row count mismatch"

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_csv, index=False)
        result["submission_path"] = str(output_csv)
        result["status"]          = "done"
        print(f"\n  Submission → {output_csv}")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n  ERROR in {name}:\n{tb}")
        result["status"]    = "error"
        result["error"]     = str(e)
        result["traceback"] = tb
        result["oof_mae"]   = float("nan")

    result["elapsed_min"] = round((time.time() - t0) / 60, 2)
    with open(result_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Result → {result_json}")
    print(f"  OOF MAE: {result.get('oof_mae', 'N/A')}")
    print(f"  Elapsed: {result['elapsed_min']:.1f} min")


if __name__ == "__main__":
    main()
