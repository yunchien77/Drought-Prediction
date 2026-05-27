import json
import numpy as np
from pathlib import Path

from config import (
    METEO_COLS, COL_IDX, N_METEO,
    WINDOW_SIZES, TREND_COLS, FEATURE_NAMES_PATH,
    SCORE_ACF_BASE,
)
from proxy import get_proxy_scores


def build_feature_names() -> list[str]:
    names = []
    for w in WINDOW_SIZES:
        for col in METEO_COLS:
            names += [f"{col}_mean_{w}d", f"{col}_std_{w}d"]
        if w in (7, 14, 21):
            for col in METEO_COLS:
                names += [f"{col}_min_{w}d", f"{col}_max_{w}d", f"{col}_p95_{w}d"]
        for col in TREND_COLS:
            names.append(f"{col}_slope_{w}d")

    for w in WINDOW_SIZES:
        for col in METEO_COLS:
            names.append(f"{col}_anom_{w}d")

    for col in METEO_COLS:
        names.append(f"{col}_mo_anom_7d")

    phys_feats = ["dry_day_frac", "heat_dd", "vpd_mean", "vpd_max",
                  "hum_aridity_idx", "prec_deficit", "dtr_mean", "prec_sum"]
    for w in WINDOW_SIZES:
        for pf in phys_feats:
            names.append(f"{pf}_{w}d")

    names += ["proxy_p7", "proxy_p21", "proxy_p91", "proxy_main",
              "proxy_7_91", "proxy_21_91", "proxy_7_21"]
    names += ["reg_mean", "reg_std", "reg_q25", "reg_q75", "reg_q90", "reg_nonzero",
              "reg_score_mo_mean", "fw_score_mo_mean"]
    names += ["sin_doy", "cos_doy", "sin_month", "cos_month", "month_raw", "quarter"]
    names += ["forecast_week", "fw_sin", "fw_cos"]

    # ── Original lag features ─────────────────────────────────────────────────
    names += ["last_known_score", "gap_weeks", "score_lag_decayed"]

    # ── NEW: extended score lag features ─────────────────────────────────────
    names += [
        "last_score_lag1",           # score 1 week before last observation
        "last_score_lag2",           # score 2 weeks before
        "last_score_lag4",           # score 4 weeks before
        "score_ewma_4w",             # exponential MA over last 4 scored weeks
        "score_trend_8w",            # linear slope over last 8 scored weeks
        "score_consecutive_nonzero", # drought persistence: consecutive non-zero weeks
        "score_lag1_decayed",        # lag1 * ACF^(gap-1)
        "score_lag2_decayed",        # lag2 * ACF^(gap-2)
        "gap_weeks_sqrt",            # sqrt transform of gap (diminishing returns)
        "gap_weeks_log1p",           # log1p transform of gap
    ]

    names += ["month_dist_to_test", "is_test_season"]
    return names


def save_feature_names(names: list[str], path: Path = FEATURE_NAMES_PATH):
    with open(path, "w") as f:
        json.dump(names, f)


def load_feature_names(path: Path = FEATURE_NAMES_PATH) -> list[str]:
    with open(path) as f:
        return json.load(f)


def make_features(
    window_arr:    np.ndarray,
    window_months: np.ndarray,
    rb:            dict,
    rm_df,
    ss:            dict,
    smo:           dict,
    forecast_week: int,
    proxy_ridge,
    last_known_score: float = float("nan"),
    gap_weeks:        float = 52.0,
    test_month:       int   = 0,
) -> np.ndarray:
    """Build a (N_FEATURES,) float32 feature vector; NaN/Inf replaced with 0."""
    feats = []
    n = len(window_arr)
    last_month = int(window_months[-1]) if n > 0 else 6

    # ── 1. Multi-scale rolling statistics ────────────────────────────────────
    window_means = {}
    for w in WINDOW_SIZES:
        tail = window_arr[-w:] if n >= w else window_arr
        m = np.nanmean(tail, axis=0)
        s = np.nanstd(tail, axis=0)
        feats.extend(m.tolist())
        feats.extend(s.tolist())
        window_means[w] = m

        if w in (7, 14, 21):
            feats.extend(np.nanmin(tail, axis=0).tolist())
            feats.extend(np.nanmax(tail, axis=0).tolist())
            feats.extend(np.nanpercentile(tail, 95, axis=0).tolist())

        for col in TREND_COLS:
            ci   = COL_IDX[col]
            vals = tail[:, ci]
            valid = ~np.isnan(vals)
            if valid.sum() > 2:
                x = np.where(valid)[0].astype(np.float32)
                slope = float(np.polyfit(x, vals[valid], 1)[0])
            else:
                slope = 0.0
            feats.append(slope)

    # ── 2. Anomaly z-scores (current window vs region long-term baseline) ────
    for w in WINDOW_SIZES:
        cur = window_means[w]
        for ci, col in enumerate(METEO_COLS):
            mu  = rb.get(f"{col}_mean", 0.0)
            sig = rb.get(f"{col}_std",  1.0)
            if not sig or np.isnan(sig) or sig < 1e-6:
                sig = 1.0
            feats.append(float((cur[ci] - mu) / sig))

    # ── 3. Monthly anomaly (current 7d vs same-month historical mean) ────────
    if rm_df is not None and last_month in rm_df.index:
        mo_row = rm_df.loc[last_month]
        for ci, col in enumerate(METEO_COLS):
            mu_mo   = mo_row.get(f"{col}_mo_mean", np.nan)
            sig_all = rb.get(f"{col}_std", 1.0)
            if not sig_all or np.isnan(sig_all) or sig_all < 1e-6:
                sig_all = 1.0
            cur7 = window_means[7][ci]
            feats.append(
                float((cur7 - mu_mo) / sig_all) if not np.isnan(mu_mo) else 0.0
            )
    else:
        feats.extend([0.0] * N_METEO)

    # ── 4. Physical drought indices ───────────────────────────────────────────
    ci_prec = COL_IDX["prec"]
    ci_tmp  = COL_IDX["tmp"]
    ci_wb   = COL_IDX["wb_tmp"]
    ci_hum  = COL_IDX["humidity"]
    ci_tmax = COL_IDX["tmp_max"]
    ci_tmin = COL_IDX["tmp_min"]
    prec_mu = float(rb.get("prec_mean", 0.0))

    for w in WINDOW_SIZES:
        tail  = window_arr[-w:] if n >= w else window_arr
        prec  = tail[:, ci_prec]
        tmp_  = tail[:, ci_tmp]
        wb_   = tail[:, ci_wb]
        hum   = tail[:, ci_hum]
        tmax  = tail[:, ci_tmax]
        tmin  = tail[:, ci_tmin]
        vpd   = tmp_ - wb_
        dtr   = tmax - tmin
        feats.append(float(np.nanmean(prec < 0.1)))
        feats.append(float(np.nanmean(np.maximum(tmp_ - 10, 0))))
        feats.append(float(np.nanmean(vpd)))
        feats.append(float(np.nanmax(vpd)))
        feats.append(float(np.nanmean(100.0 - hum)))
        feats.append(float(prec_mu * len(tail) - np.nansum(prec)))
        feats.append(float(np.nanmean(dtr)))
        feats.append(float(np.nansum(prec)))

    # ── 5. Proxy scores ───────────────────────────────────────────────────────
    p7, p21, p91, p_main = get_proxy_scores(window_arr, rb, proxy_ridge)
    feats += [p7, p21, p91, p_main, p7 - p91, p21 - p91, p7 - p21]

    # ── 6. Region score statistics (fixed, non-temporal) ─────────────────────
    feats.append(float(ss.get("reg_mean",    0.5)))
    feats.append(float(ss.get("reg_std",     0.5)))
    feats.append(float(ss.get("reg_q25",     0.0)))
    feats.append(float(ss.get("reg_q75",     1.0)))
    feats.append(float(ss.get("reg_q90",     2.0)))
    feats.append(float(ss.get("reg_nonzero", 0.4)))
    mo_score_mean = smo.get(last_month, ss.get("reg_mean", 0.5))
    feats.append(float(mo_score_mean))
    fw_month = ((last_month - 1 + (forecast_week - 1) // 4) % 12) + 1
    feats.append(float(smo.get(fw_month, ss.get("reg_mean", 0.5))))

    # ── 7. Seasonal encoding ──────────────────────────────────────────────────
    doy = last_month * 30
    feats.append(float(np.sin(2 * np.pi * doy / 365)))
    feats.append(float(np.cos(2 * np.pi * doy / 365)))
    feats.append(float(np.sin(2 * np.pi * last_month / 12)))
    feats.append(float(np.cos(2 * np.pi * last_month / 12)))
    feats.append(float(last_month))
    feats.append(float((last_month - 1) // 3 + 1))

    # ── 8. Forecast week encoding ─────────────────────────────────────────────
    feats.append(float(forecast_week))
    feats.append(float(np.sin(2 * np.pi * forecast_week / 5)))
    feats.append(float(np.cos(2 * np.pi * forecast_week / 5)))

    # ── 9. Score lag features ─────────────────────────────────────────────────
    reg_mean = float(ss.get("reg_mean", 0.5))
    lks = last_known_score if np.isfinite(last_known_score) else reg_mean
    gw  = gap_weeks        if np.isfinite(gap_weeks)        else 52.0

    feats.append(float(lks))
    feats.append(float(gw))
    feats.append(float(lks * SCORE_ACF_BASE ** gw))

    # ── NEW: extended score lag features ─────────────────────────────────────
    # Read from ss (populated by climatology.py at region level).
    # During training, these reflect the state at the training window's cutoff;
    # during inference they reflect the last known state from train data.
    l1 = float(ss.get("last_score_lag1", lks))
    l2 = float(ss.get("last_score_lag2", lks))
    l4 = float(ss.get("last_score_lag4", lks))
    if not np.isfinite(l1): l1 = lks
    if not np.isfinite(l2): l2 = lks
    if not np.isfinite(l4): l4 = lks

    ewma   = float(ss.get("score_ewma_4w",             lks))
    trend  = float(ss.get("score_trend_8w",             0.0))
    consec = float(ss.get("score_consecutive_nonzero",  0.0))
    if not np.isfinite(ewma):   ewma   = lks
    if not np.isfinite(trend):  trend  = 0.0
    if not np.isfinite(consec): consec = 0.0

    feats.append(l1)
    feats.append(l2)
    feats.append(l4)
    feats.append(ewma)
    feats.append(trend)
    feats.append(consec)

    # Decayed versions of lag1 / lag2
    feats.append(float(l1 * SCORE_ACF_BASE ** max(gw - 1.0, 0.0)))
    feats.append(float(l2 * SCORE_ACF_BASE ** max(gw - 2.0, 0.0)))

    # Gap monotone transforms (help the tree find the right split boundary)
    feats.append(float(np.sqrt(max(gw, 0.0))))
    feats.append(float(np.log1p(max(gw, 0.0))))

    # ── 10. Test season distance features ────────────────────────────────────
    if test_month > 0:
        dist = abs(last_month - test_month)
        dist = min(dist, 12 - dist)
    else:
        dist = 3
    feats.append(float(dist))
    feats.append(float(1 if dist <= 2 else 0))

    arr = np.array(feats, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
