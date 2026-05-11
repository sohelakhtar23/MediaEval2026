"""
clip_merge_experiment.py
========================
Loads the already-computed clip_features.csv and merges it into the
pipeline_v4 ablation framework as a 5th feature stream.

Strategy:
  - Runs Spearman correlation of each CLIP feature vs. both targets.
  - Tries three merge configurations and reports CV SRCC for each:
      A) CLIP all 9 features added to the full pipeline_v4 feature set
      B) Only bmem_brand_specific added to CNN-32 (brand model candidate)
      C) Only CLIP features with |r| >= CORR_THRESHOLD added to full set
  - Does NOT touch pipeline_v4.py — imports helpers from it directly.

Run after pipeline_v4.py has been run at least once so that
llm_scalar_cache.json exists (avoids re-calling OpenAI).
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Import everything we need from pipeline_v4
from pipeline_v4 import (
    DATA_ROOT, STT_DIR, FRAMES_DIR, FEATURES_DIR, CACHE_FILE,
    CNN_MODELS, RANDOM_SEED, N_FOLDS,
    extract_features, apply_cnn_pca, _to_array,
    evaluate_model, print_results,
)
from openai import OpenAI

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

CLIP_CSV        = Path("clip_features.csv")
CORR_THRESHOLD  = 0.10      # min |Spearman r| for a CLIP feature to be included in config C

CLIP_FEATURES = [
    "vmem_mean", "vmem_max", "vmem_std", "vmem_slope",
    "bmem_mean", "bmem_max", "bmem_std", "bmem_slope",
    "bmem_brand_specific",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def cv_loop(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            weights: np.ndarray, model_name: str = "RF") -> float:
    """Single-target 5-fold GroupKFold. Returns mean SRCC."""
    gkf    = GroupKFold(n_splits=N_FOLDS)
    scores = []

    for train_idx, val_idx in gkf.split(X, groups=groups):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr        = weights[train_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        mdl = (BayesianRidge() if model_name == "BayesianRidge"
               else RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                          random_state=RANDOM_SEED))
        mdl.fit(X_tr, y_tr, sample_weight=w_tr)
        preds = mdl.predict(X_val)
        scores.append(spearmanr(y_val, preds).correlation)

    return float(np.mean(scores))


def print_corr_table(clip_df: pd.DataFrame,
                     y_mem: np.ndarray, y_brand: np.ndarray):
    bar = "─" * 58
    print(f"\n{bar}")
    print(f"  {'CLIP feature':<28}  {'mem r':>8}  {'brand r':>8}")
    print(bar)
    for col in CLIP_FEATURES:
        r_mem,   _ = spearmanr(clip_df[col], y_mem)
        r_brand, _ = spearmanr(clip_df[col], y_brand)
        flag = " ◄" if (abs(r_mem) >= CORR_THRESHOLD or
                        abs(r_brand) >= CORR_THRESHOLD) else ""
        print(f"  {col:<28}  {r_mem:>+8.3f}  {r_brand:>+8.3f}{flag}")
    print(f"{bar}")
    print(f"  ◄ = |r| ≥ {CORR_THRESHOLD} (included in config C)\n")


def compare(label: str, X_mem: np.ndarray, X_brand: np.ndarray,
            y_mem, y_brand, groups, weights):
    """Run RF and BayesianRidge for both targets, print one summary line."""
    best_mem   = max(cv_loop(X_mem,   y_mem,   groups, weights, "RF"),
                     cv_loop(X_mem,   y_mem,   groups, weights, "BayesianRidge"))
    best_brand = max(cv_loop(X_brand, y_brand, groups, weights, "RF"),
                     cv_loop(X_brand, y_brand, groups, weights, "BayesianRidge"))
    print(f"  {label:<44}  mem={best_mem:+.3f}  brand={best_brand:+.3f}")
    return best_mem, best_brand


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load metadata + targets ──────────────────────────────────────────────
    df      = pd.read_csv(DATA_ROOT / "devset_videolist_GT.csv")
    y_mem   = df["memorability_score"].values.astype(float)
    y_brand = df["brand_memorability"].values.astype(float)
    groups  = df["channelName"].values
    weights = df["nb_annotations"].values.astype(float)
    weights = weights / weights.mean()

    # ── Load CLIP features ───────────────────────────────────────────────────
    if not CLIP_CSV.exists():
        raise FileNotFoundError(f"{CLIP_CSV} not found — run clip_features.py first.")

    clip_df = pd.read_csv(CLIP_CSV)
    clip_df = clip_df.merge(df[["id"]], on="id", how="right")   # align rows with df
    print(f"CLIP features loaded: {len(clip_df)} rows, "
          f"{clip_df[CLIP_FEATURES].isnull().sum().sum()} NaNs")
    clip_df[CLIP_FEATURES] = clip_df[CLIP_FEATURES].fillna(
        clip_df[CLIP_FEATURES].mean())

    X_clip     = clip_df[CLIP_FEATURES].values.astype(float)
    X_clip_std = StandardScaler().fit_transform(X_clip)

    # ── Spearman correlation table ───────────────────────────────────────────
    print_corr_table(clip_df, y_mem, y_brand)

    # ── Re-extract pipeline_v4 features (uses LLM cache — no API calls) ─────
    print("Re-extracting pipeline_v4 features (LLM cache will be used) ...")
    api_key       = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=api_key) if api_key else None

    cache: dict = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as fh:
            cache = json.load(fh)
        print(f"  LLM cache: {len(cache)} entries (no new API calls expected)")

    frame_df, brand_df, llm_df, cnn_raw = extract_features(
        df, STT_DIR, FRAMES_DIR, cache, openai_client, FEATURES_DIR)

    X_cnn, _, _ = apply_cnn_pca(cnn_raw)
    X_frame      = _to_array(frame_df)
    X_brand_feat = _to_array(brand_df)
    X_llm        = _to_array(llm_df)

    # Full v4 feature matrix (62 features — baseline)
    X_full_v4 = np.concatenate([X_frame, X_brand_feat, X_llm, X_cnn], axis=1)

    # ── Determine config C: CLIP features with |r| >= threshold ─────────────
    selected = [col for col in CLIP_FEATURES
                if (abs(spearmanr(clip_df[col], y_mem).correlation)   >= CORR_THRESHOLD or
                    abs(spearmanr(clip_df[col], y_brand).correlation) >= CORR_THRESHOLD)]
    print(f"Config C selected features (|r| ≥ {CORR_THRESHOLD}): "
          f"{selected if selected else 'none'}")

    X_clip_selected = (clip_df[selected].values.astype(float)
                       if selected else np.zeros((len(df), 0)))

    # ── Comparison table ─────────────────────────────────────────────────────
    bar = "═" * 70
    print(f"\n{bar}")
    print("  MERGE EXPERIMENT — best of (RF, BayesianRidge) per config")
    print(f"  Baseline numbers from pipeline_v4 (GroupKFold, N=339)")
    print(f"{'─'*70}")
    print(f"  {'Config':<44}  {'mem SRCC':>8}  {'brand SRCC':>10}")
    print(f"{'─'*70}")

    # Baselines
    compare("BASELINE: pipeline_v4 full (62 feats)",
            X_full_v4, X_cnn, y_mem, y_brand, groups, weights)
    compare("BASELINE: CLIP only (9 feats)",
            X_clip_std, X_clip_std, y_mem, y_brand, groups, weights)

    print(f"{'─'*70}")

    # Config A — all 9 CLIP added to full v4
    X_A = np.concatenate([X_full_v4, X_clip_std], axis=1)
    compare("A: v4_full + CLIP all 9",
            X_A, X_A, y_mem, y_brand, groups, weights)

    # Config B — only bmem_brand_specific added to CNN-32 (brand model)
    X_brand_specific = clip_df[["bmem_brand_specific"]].values.astype(float)
    X_B_brand        = np.concatenate([X_cnn, X_brand_specific], axis=1)
    compare("B: CNN-32 + bmem_brand_specific (brand only)",
            X_full_v4,   # mem stays the same
            X_B_brand,
            y_mem, y_brand, groups, weights)

    # Config C — only high-correlation CLIP features
    if X_clip_selected.shape[1] > 0:
        X_C = np.concatenate([X_full_v4, X_clip_selected], axis=1)
        compare(f"C: v4_full + {len(selected)} selected CLIP feats",
                X_C, X_C, y_mem, y_brand, groups, weights)
    else:
        print(f"  {'C: no CLIP features passed threshold — skipped':<68}")

    print(f"{bar}\n")
    print("Interpretation guide:")
    print("  • If A > baseline: CLIP adds signal even for video mem.")
    print("  • If B brand > baseline brand: bmem_brand_specific helps CNN-32.")
    print("  • If C > A: selective CLIP is better than all 9.")
    print("  Use the winning config to update pipeline_v4's feature assembly.")


if __name__ == "__main__":
    main()
