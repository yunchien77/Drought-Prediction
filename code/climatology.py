import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from config import METEO_COLS, CLIMATOLOGY_PATH
from logging_setup import get_logger

log = get_logger("climatology")


# ─────────────────────────────────────────────────────────────────────────────
# Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_climatology(train: pd.DataFrame) -> dict:
    """Compute four types of per-region statistics from the training DataFrame.

    Requires a 'month' column (int8) to be present in train.
    Returns a dict with keys: clim_dict, month_dict, sstat_dict, smo_dict.
    """
    log.info("Computing region climatology baselines ...")

    # 1. Long-term mean / std (meteorological only)
    clim_stats = (
        train.groupby("region_id")[METEO_COLS]
        .agg(["mean", "std"])
        .astype("float32")
    )
    clim_stats.columns = [f"{c}_{s}" for c, s in clim_stats.columns]
    clim_dict = {
        r: row.to_dict()
        for r, row in clim_stats.iterrows()
    }

    # 2. Monthly means (seasonal baseline)
    monthly_clim = (
        train.groupby(["region_id", "month"])[METEO_COLS]
        .mean()
        .astype("float32")
        .reset_index()
    )
    monthly_clim.columns = (
        ["region_id", "month"] + [f"{c}_mo_mean" for c in METEO_COLS]
    )
    month_dict = {
        r: grp.set_index("month")
        for r, grp in monthly_clim.groupby("region_id")
    }

    # 3. Per-region score statistics (fixed, non-temporal)
    scored = train.dropna(subset=["score"])
    reg_score_stats = (
        scored.groupby("region_id")["score"]
        .agg(
            reg_mean="mean",
            reg_std="std",
            reg_q25=lambda x: float(x.quantile(0.25)),
            reg_q75=lambda x: float(x.quantile(0.75)),
            reg_q90=lambda x: float(x.quantile(0.90)),
            reg_nonzero=lambda x: float((x > 0).mean()),
        )
        .reset_index()
    )
    sstat_dict = {
        r: row.to_dict()
        for r, row in reg_score_stats.set_index("region_id").iterrows()
    }

    # 3b. Last known score per region (used as score lag feature)
    def _dt_to_ord(s: str) -> int:
        """Manual Gregorian ordinal that handles years > 9999."""
        try:
            parts = str(s).strip().split("T")[0].split(" ")[0]
            p = parts.replace("/", "-").replace(".", "-").split("-")
            y, m, d = int(p[0]), int(p[1]), int(p[2])
            is_leap = (y % 4 == 0) and (y % 100 != 0 or y % 400 == 0)
            mdays = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            doy = sum(mdays[1:m]) + (1 if m > 2 and is_leap else 0) + d
            y1 = y - 1
            return y1 * 365 + y1 // 4 - y1 // 100 + y1 // 400 + doy
        except Exception:
            return 0

    for region, grp in scored.groupby("region_id"):
        scores_arr = grp["score"].values.astype(np.float32)
        dates_arr  = grp["date"].values
        n = len(scores_arr)

        last_row = grp.iloc[-1]
        last_sc  = float(last_row["score"])
        last_dt  = str(last_row["date"])
        last_ord = _dt_to_ord(last_dt)

        if region not in sstat_dict:
            continue

        sstat_dict[region]["last_score"]        = last_sc
        sstat_dict[region]["last_date_ordinal"] = last_ord

        # ── NEW: extended score lag features ─────────────────────────────────
        # lag1 / lag2 / lag4 (weekly lags, fall back to last_sc if not enough data)
        sstat_dict[region]["last_score_lag1"] = float(scores_arr[-1]) if n >= 1 else last_sc
        sstat_dict[region]["last_score_lag2"] = float(scores_arr[-2]) if n >= 2 else last_sc
        sstat_dict[region]["last_score_lag4"] = float(scores_arr[-4]) if n >= 4 else last_sc

        # EWMA over last 4 weeks (heavier weight on most recent)
        k = min(n, 4)
        weights = np.array([0.5 ** (k - 1 - i) for i in range(k)], dtype=np.float32)
        weights /= weights.sum()
        sstat_dict[region]["score_ewma_4w"] = float(np.dot(scores_arr[-k:], weights))

        # Linear trend slope over last 8 scored weeks
        k8 = min(n, 8)
        if k8 >= 3:
            x_    = np.arange(k8, dtype=np.float32)
            slope = float(np.polyfit(x_, scores_arr[-k8:], 1)[0])
        else:
            slope = 0.0
        sstat_dict[region]["score_trend_8w"] = slope

        # Consecutive non-zero weeks immediately before end
        consec = 0
        for sc in reversed(scores_arr):
            if sc > 0:
                consec += 1
            else:
                break
        sstat_dict[region]["score_consecutive_nonzero"] = float(consec)

    # 4. Monthly score means
    reg_score_monthly = (
        scored.groupby(["region_id", "month"])["score"]
        .mean()
        .rename("reg_score_mo_mean")
        .reset_index()
    )
    smo_dict = {
        r: grp.set_index("month")["reg_score_mo_mean"].to_dict()
        for r, grp in reg_score_monthly.groupby("region_id")
    }

    log.info(f"  clim_dict: {len(clim_dict)} regions")
    log.info(f"  month_dict: {len(month_dict)} regions")
    log.info(f"  sstat_dict: {len(sstat_dict)} regions")
    log.info(f"  smo_dict: {len(smo_dict)} regions")

    return dict(
        clim_dict=clim_dict,
        month_dict=month_dict,
        sstat_dict=sstat_dict,
        smo_dict=smo_dict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_climatology(clim: dict, path: Path = CLIMATOLOGY_PATH):
    with open(path, "wb") as f:
        pickle.dump(clim, f, protocol=4)
    log.info(f"climatology saved → {path}")


def load_climatology(path: Path = CLIMATOLOGY_PATH) -> dict:
    with open(path, "rb") as f:
        clim = pickle.load(f)
    log.info(f"climatology loaded ← {path}")
    return clim
