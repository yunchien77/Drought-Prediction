"""DLinear time-series model for drought score forecasting.

Architecture (Zeng et al. 2023 "Are Transformers Effective for Time Series?"):
  - Decompose each input feature into trend + seasonal via moving average
  - Independent Linear projection per feature  (trend: seq→1, seasonal: seq→1)
  - Final linear mixing across all features → scalar drought-score prediction

Integration:
  - Stage 5 in train.py  → builds raw-meteo windows, trains DLinear, saves OOF MAE
  - predict.py           → loads DLinear, blends with LightGBM predictions

Feature windows mirror the LightGBM data pipeline:
  For anchor score at day t, forecast horizon fw:
      window ends at  w_end   = t - (fw - 1) * 7
      window starts at w_start = max(0, w_end - SEQ_LEN)
  This ensures DLinear sees the same information state as LightGBM.

Input tensor shape:  (B, SEQ_LEN, N_FEATURES)
  N_FEATURES = len(METEO_COLS) + 1  (last_known_score appended as final channel)
  SEQ_LEN    = DLINEAR_SEQ_LEN  (default 91 days)
Output shape:        (B, 1)
"""
from __future__ import annotations

import gc
import time
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from logging_setup import get_logger

log = get_logger("dlinear")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy torch import — graceful degradation if PyTorch not installed
# ─────────────────────────────────────────────────────────────────────────────

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for DLinear.  "
            "Install it with:  pip install torch>=2.0.0\n"
            "Or disable the model:  set USE_DLINEAR=False in config.py"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

class _MovingAvg:
    """Numpy moving average for trend extraction (used during inference too)."""
    def __init__(self, kernel_size: int):
        self.k = kernel_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (T, C)  → smooth along T per channel
        pad_front = x[:1, :].repeat((self.k - 1) // 2, axis=0)
        pad_end   = x[-1:, :].repeat((self.k - 1) // 2, axis=0)
        padded    = np.concatenate([pad_front, x, pad_end], axis=0)
        T, C = x.shape
        out = np.zeros_like(x)
        for c in range(C):
            out[:, c] = np.convolve(padded[:, c], np.ones(self.k) / self.k, mode='valid')[:T]
        return out


def _build_model_class():
    torch, nn = _require_torch()

    class _MovAvgTorch(nn.Module):
        def __init__(self, kernel_size: int):
            super().__init__()
            self.k   = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, C)
            front = x[:, :1, :].repeat(1, (self.k - 1) // 2, 1)
            end   = x[:, -1:, :].repeat(1, (self.k - 1) // 2, 1)
            padded = torch.cat([front, x, end], dim=1)
            # pool over T per channel: permute to (B, C, T)
            trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
            return trend

    class DLinearNet(nn.Module):
        """Lightweight DLinear regressor.

        Each of the C input channels gets its own pair of linear projections
        (trend + seasonal, seq_len → 1). The C scalar outputs are then mixed
        by a final learnable linear layer into a single drought-score prediction.
        """
        def __init__(self, seq_len: int, n_features: int, kernel_size: int = 25):
            super().__init__()
            self.seq_len   = seq_len
            self.n_features = n_features
            self.decomp    = _MovAvgTorch(kernel_size)

            # Per-feature projections: weight shape (n_features, seq_len)
            self.trend_w    = nn.Parameter(torch.zeros(n_features, seq_len))
            self.seasonal_w = nn.Parameter(torch.zeros(n_features, seq_len))
            nn.init.xavier_uniform_(self.trend_w.unsqueeze(0))
            nn.init.xavier_uniform_(self.seasonal_w.unsqueeze(0))

            # Mix C feature contributions into one output
            self.mix = nn.Linear(n_features, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, C)
            trend    = self.decomp(x)              # (B, T, C)
            seasonal = x - trend                   # (B, T, C)

            # Per-feature dot product: (B, T, C) × (C, T) → einsum → (B, C)
            t_out = torch.einsum("btc,ct->bc", trend,    self.trend_w)    # (B, C)
            s_out = torch.einsum("btc,ct->bc", seasonal, self.seasonal_w) # (B, C)

            combined = t_out + s_out               # (B, C)
            out      = self.mix(combined)          # (B, 1)
            return torch.clamp(out, 0.0, 5.0)

    return DLinearNet


# ─────────────────────────────────────────────────────────────────────────────
# Data building (mirrors data_pipeline._worker_train_region window logic)
# ─────────────────────────────────────────────────────────────────────────────

def _date_to_ord(s) -> int:
    """Fast date-string → ordinal integer (supports years > 9999)."""
    try:
        parts = str(s).strip().split("T")[0].split(" ")[0].split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        _mdays  = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        is_leap = (y % 4 == 0) and (y % 100 != 0 or y % 400 == 0)
        doy = sum(_mdays[1:m]) + (1 if m > 2 and is_leap else 0) + d
        y1  = y - 1
        return y1 * 365 + y1 // 4 - y1 // 100 + y1 // 400 + doy
    except Exception:
        return 0


def build_dlinear_dataset(
    train: pd.DataFrame,
    meteo_cols: list[str],
    seq_len: int,
    n_pred: int = 5,
    seed: int = 42,
    max_win: int = 15,
    min_gap_days: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build raw-window training dataset for DLinear.

    Memory-safe design:
    - max_win=15 per region → 2248 × 15 × 5 ≈ 168k samples (~0.6 GB)
    - min_gap_days: minimum gap between window-end and target to prevent leakage
      (14 days ensures the last_score feature is ≥2 weeks old at prediction time,
       matching the realistic test scenario where train-test gap is weeks/months)

    Returns
    -------
    X_raw  : (N, seq_len, n_features)  float32
    y      : (N,)                       float32
    tk     : (N,)                       int64    — for TemporalKFold
    fw     : (N,)                       int8
    groups : (N,)                       object
    """
    n_meteo    = len(meteo_cols)
    n_features = n_meteo + 1     # meteo channels + last_known_score

    X_list, y_list, tk_list, fw_list, grp_list = [], [], [], [], []

    for region, grp in train.groupby("region_id"):
        grp    = grp.sort_values("date").reset_index(drop=True)
        meteo  = grp[meteo_cols].values.astype(np.float32)
        scores = grp["score"].values.astype(np.float32)
        dates  = grp["date"].values

        score_idxs = np.where(~np.isnan(scores))[0]
        if len(score_idxs) < 10:
            continue

        # Subsample: keep at most max_win anchors (bias toward recent)
        rng = np.random.RandomState(seed + hash(str(region)) % 100000)
        if len(score_idxs) > max_win:
            split    = int(len(score_idxs) * 0.6)
            recent   = score_idxs[split:]
            early    = score_idxs[:split]
            n_recent = min(max_win, len(recent))
            n_early  = max_win - n_recent
            chosen   = list(rng.choice(recent, n_recent, replace=False))
            if n_early > 0 and len(early) > 0:
                chosen += list(rng.choice(early, min(n_early, len(early)), replace=False))
        else:
            chosen = score_idxs.tolist()

        date_ords = np.array([_date_to_ord(d) for d in dates], dtype=np.int64)

        for target_idx in chosen:
            for fw in range(1, n_pred + 1):
                # Window ends (fw-1)*7 days before target — same as LightGBM
                w_end   = max(7, target_idx - (fw - 1) * 7)
                w_start = max(0, w_end - seq_len)
                win     = meteo[w_start:w_end]

                # Anti-leakage: last_score must be ≥ min_gap_days before target.
                # This simulates the realistic test scenario where the most recent
                # known score is weeks old, not days old.
                # Without this, model learns trivial autocorrelation (OOF MAE ~0.19)
                # that doesn't generalise to the test gap of weeks/months.
                safe_end  = max(0, w_end - min_gap_days)
                valid_sc  = np.where(~np.isnan(scores[:safe_end]))[0]
                last_sc   = float(scores[valid_sc[-1]]) if len(valid_sc) > 0 else 0.5

                # Pad to exactly seq_len
                pad_len = seq_len - len(win)
                if pad_len > 0:
                    win = np.concatenate(
                        [np.zeros((pad_len, n_meteo), dtype=np.float32), win], axis=0
                    )
                else:
                    win = win[-seq_len:]

                last_sc_ch = np.full((seq_len, 1), last_sc, dtype=np.float32)
                window = np.concatenate([win, last_sc_ch], axis=1)   # (seq_len, n_features)

                X_list.append(window)
                y_list.append(float(scores[target_idx]))

                t_key = int(date_ords[target_idx])
                if t_key == 0:
                    nz = date_ords[max(0, target_idx - 14):target_idx + 1]
                    t_key = int(nz[nz > 0][-1]) if (nz > 0).any() else 0
                tk_list.append(t_key)
                fw_list.append(fw)
                grp_list.append(region)

    X      = np.stack(X_list,  axis=0).astype(np.float32)
    y      = np.array(y_list,  dtype=np.float32)
    tk     = np.array(tk_list, dtype=np.int64)
    fw_arr = np.array(fw_list, dtype=np.int8)
    groups = np.array(grp_list)

    mem_gb = X.nbytes / 1e9
    log.info(f"  DLinear dataset: {X.shape[0]:,} samples  "
             f"seq={seq_len}  feats={X.shape[2]}  RAM≈{mem_gb:.1f}GB  "
             f"score=[{y.min():.2f},{y.max():.2f}]")
    return X, y, tk, fw_arr, groups


# ─────────────────────────────────────────────────────────────────────────────
# Feature normalisation (per-channel StandardScaler across time+batch)
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from (N, T, C) array."""
    N, T, C = X.shape
    flat   = X.reshape(-1, C)          # (N*T, C)
    mean_  = flat.mean(axis=0)         # (C,)
    std_   = flat.std(axis=0) + 1e-8   # (C,)
    return mean_, std_


def apply_scaler(X: np.ndarray, mean_: np.ndarray, std_: np.ndarray) -> np.ndarray:
    return (X - mean_) / std_


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_dlinear(train: pd.DataFrame) -> float:
    """Train DLinear model and save artifacts.  Returns OOF MAE."""
    torch, nn = _require_torch()
    from sklearn.metrics import mean_absolute_error as _mae

    from config import (
        METEO_COLS, N_PRED, SEED,
        DLINEAR_SEQ_LEN, DLINEAR_EPOCHS, DLINEAR_BATCH_SIZE,
        DLINEAR_LR, DLINEAR_KERNEL_SIZE, DLINEAR_MODEL_PATH,
        DLINEAR_MAX_WIN, PURGE_GAP_DAYS,
    )
    from temporal_cv import TemporalKFold

    t0 = time.time()
    log.info(f"[DLinear] Building training windows  "
             f"(max_win={DLINEAR_MAX_WIN}, seq={DLINEAR_SEQ_LEN}) ...")
    X, y, tk, fw_arr, groups = build_dlinear_dataset(
        train, METEO_COLS, seq_len=DLINEAR_SEQ_LEN,
        n_pred=N_PRED, seed=SEED,
        max_win=DLINEAR_MAX_WIN,
        min_gap_days=14,   # prevent leakage: last_score ≥ 14 days old
    )

    log.info("[DLinear] Fitting scaler ...")
    mean_, std_ = fit_scaler(X)
    X_norm = apply_scaler(X, mean_, std_).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[DLinear] Device: {device}")

    DLinearNet = _build_model_class()
    n_features = X.shape[2]

    # ── Walk-forward OOF ────────────────────────────────────────────────────
    tkf = TemporalKFold(n_splits=4, gap_days=PURGE_GAP_DAYS)
    oof_preds = np.full(len(y), np.nan, dtype=np.float32)
    fold_maes = []

    for fold_idx, (tr_pos, va_pos) in enumerate(tkf.split(tk)):
        if len(tr_pos) < 50 or len(va_pos) < 10:
            continue
        log.info(f"  Fold {fold_idx+1}  tr={len(tr_pos):,}  va={len(va_pos):,}")

        model = DLinearNet(DLINEAR_SEQ_LEN, n_features, DLINEAR_KERNEL_SIZE).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=DLINEAR_LR)
        loss_fn = nn.L1Loss()

        import torch.utils.data as data_utils
        tr_ds = data_utils.TensorDataset(
            torch.tensor(X_norm[tr_pos], dtype=torch.float32),
            torch.tensor(y[tr_pos],      dtype=torch.float32),
        )
        tr_loader = data_utils.DataLoader(
            tr_ds, batch_size=DLINEAR_BATCH_SIZE, shuffle=True,
            drop_last=False, num_workers=0,
        )

        model.train()
        for ep in range(DLINEAR_EPOCHS):
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()

        # Validation
        model.eval()
        Xva = torch.tensor(X_norm[va_pos], dtype=torch.float32).to(device)
        with torch.no_grad():
            va_pred = model(Xva).squeeze(-1).cpu().numpy()
        va_pred = np.clip(va_pred, 0, 5)
        oof_preds[va_pos] = va_pred
        fold_maes.append(float(_mae(y[va_pos], va_pred)))
        log.info(f"  Fold {fold_idx+1} val MAE: {fold_maes[-1]:.4f}")

    valid   = ~np.isnan(oof_preds)
    oof_mae = float(_mae(y[valid], np.clip(oof_preds[valid], 0, 5))) if valid.sum() > 0 else float("nan")
    log.info(f"[DLinear] OOF MAE: {oof_mae:.4f}  (fold MAEs: {[f'{m:.4f}' for m in fold_maes]})")

    # ── Retrain on full data ─────────────────────────────────────────────────
    log.info("[DLinear] Retraining on full dataset ...")
    model = DLinearNet(DLINEAR_SEQ_LEN, n_features, DLINEAR_KERNEL_SIZE).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=DLINEAR_LR)
    loss_fn = nn.L1Loss()

    import torch.utils.data as data_utils
    full_ds = data_utils.TensorDataset(
        torch.tensor(X_norm, dtype=torch.float32),
        torch.tensor(y,      dtype=torch.float32),
    )
    full_loader = data_utils.DataLoader(
        full_ds, batch_size=DLINEAR_BATCH_SIZE, shuffle=True,
        drop_last=False, num_workers=0,
    )
    model.train()
    for ep in range(DLINEAR_EPOCHS):
        ep_loss = 0.0
        for xb, yb in full_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        if (ep + 1) % 10 == 0:
            log.info(f"  Full retrain epoch {ep+1}/{DLINEAR_EPOCHS}  "
                     f"MAE={ep_loss/len(y):.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    DLINEAR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "mean_": mean_,
        "std_":  std_,
        "seq_len":    DLINEAR_SEQ_LEN,
        "n_features": n_features,
        "kernel_size": DLINEAR_KERNEL_SIZE,
        "oof_mae":    oof_mae,
        "meteo_cols": METEO_COLS,
    }
    torch.save(ckpt, DLINEAR_MODEL_PATH)
    log.info(f"[DLinear] Saved → {DLINEAR_MODEL_PATH}  "
             f"({(time.time()-t0)/60:.1f} min)")
    return oof_mae


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_dlinear() -> Optional[dict]:
    """Load DLinear checkpoint.  Returns None if not found or torch unavailable."""
    try:
        from config import DLINEAR_MODEL_PATH
        import torch
        if not DLINEAR_MODEL_PATH.exists():
            log.warning(f"[DLinear] Model not found at {DLINEAR_MODEL_PATH}")
            return None
        # weights_only=False required: checkpoint contains numpy arrays (scaler)
        # PyTorch ≥ 2.4 changed the default to weights_only=True which rejects numpy
        ckpt = torch.load(DLINEAR_MODEL_PATH, map_location="cpu", weights_only=False)
        log.info(f"[DLinear] Loaded checkpoint  OOF MAE: {ckpt.get('oof_mae', 'N/A')}")
        return ckpt
    except Exception as e:
        log.warning(f"[DLinear] Failed to load checkpoint ({e})", exc_info=True)
        return None


def predict_dlinear(
    test: pd.DataFrame,
    train: pd.DataFrame,
    ckpt: dict,
) -> dict[str, list[float]]:
    """Generate DLinear predictions for all test regions.

    For each region and each forecast week fw=1..5:
        window ends   at last training day - (fw-1)*7
        window starts at window end - seq_len

    Returns dict  {region_id: [fw1_pred, fw2_pred, fw3_pred, fw4_pred, fw5_pred]}
    """
    torch, nn = _require_torch()

    seq_len     = ckpt["seq_len"]
    n_features  = ckpt["n_features"]
    kernel_size = ckpt["kernel_size"]
    mean_       = ckpt["mean_"]
    std_        = ckpt["std_"]
    meteo_cols  = ckpt["meteo_cols"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DLinearNet = _build_model_class()
    model = DLinearNet(seq_len, n_features, kernel_size).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Pre-index training data by region for fast lookup
    train_by_region = {
        r: g.sort_values("date").reset_index(drop=True)
        for r, g in train.groupby("region_id")
    }

    from config import N_PRED
    preds: dict[str, list[float]] = {}
    fallback_sc = float(train["score"].dropna().mean())

    for region_id in test["region_id"].unique():
        r_key = region_id
        if r_key not in train_by_region:
            preds[r_key] = [fallback_sc] * N_PRED
            continue

        tr_reg = train_by_region[r_key]
        meteo_all = tr_reg[meteo_cols].values.astype(np.float32)   # (T, 9)
        scores_all = tr_reg["score"].values.astype(np.float32)

        valid_sc = np.where(~np.isnan(scores_all))[0]
        last_sc  = float(scores_all[valid_sc[-1]]) if len(valid_sc) > 0 else fallback_sc

        region_preds = []
        windows = []

        for fw in range(1, N_PRED + 1):
            # Mirror the train window logic: window ends (fw-1)*7 days before present
            t_end   = len(meteo_all) - (fw - 1) * 7
            t_start = max(0, t_end - seq_len)
            win     = meteo_all[t_start:t_end]

            # Pad if needed
            pad_len = seq_len - len(win)
            if pad_len > 0:
                win = np.concatenate(
                    [np.zeros((pad_len, len(meteo_cols)), dtype=np.float32), win], axis=0
                )

            # Append last_score channel
            last_sc_ch = np.full((seq_len, 1), last_sc, dtype=np.float32)
            window = np.concatenate([win, last_sc_ch], axis=1)  # (seq_len, n_features)
            windows.append(window)

        # Batch predict all 5 fw at once
        X_batch = np.stack(windows, axis=0).astype(np.float32)   # (5, seq_len, n_features)
        X_norm  = (X_batch - mean_) / std_
        X_t     = torch.tensor(X_norm, dtype=torch.float32).to(device)

        with torch.no_grad():
            batch_pred = model(X_t).squeeze(-1).cpu().numpy()   # (5,)

        batch_pred = np.clip(batch_pred, 0.0, 5.0)
        preds[r_key] = [round(float(batch_pred[fw - 1]), 4) for fw in range(1, N_PRED + 1)]

    log.info(f"[DLinear] Predicted {len(preds)} regions")
    return preds
