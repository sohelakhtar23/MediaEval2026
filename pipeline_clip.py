"""
Pipeline — Direction 1: CLIP Zero-Shot Features
================================================
Reads clip_features.csv + metadata CSV, runs 5-fold channel-stratified CV,
tries RF and Ridge/SVR per target, reports mean Spearman SRCC.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────

CLIP_CSV     = "clip_features.csv"
METADATA_CSV = "devset_videolist_GT.csv"

TARGET_VIDEO = "memorability_score"
TARGET_BRAND = "brand_memorability"

N_FOLDS      = 5
RANDOM_STATE = 42

# ── Feature columns (all CLIP-derived) ───────────────────────────────────────

CLIP_FEATURES = [
    "vmem_mean", "vmem_max", "vmem_std", "vmem_slope",
    "bmem_mean", "bmem_max", "bmem_std", "bmem_slope",
    "bmem_brand_specific",
]

# ── Models to compare ─────────────────────────────────────────────────────────
#
# Each model is a sklearn Pipeline so StandardScaler is applied where needed.
# RF does not need scaling; Ridge and SVR do.

def build_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=10.0)),
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel="rbf", C=1.0, epsilon=0.05)),
        ]),
    }


# ── CV loop ───────────────────────────────────────────────────────────────────

def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, target_name: str):
    """
    5-fold GroupKFold CV.
    Returns a dict: model_name → mean SRCC over folds.
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    models = build_models()
    results = {name: [] for name in models}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            srcc, _ = spearmanr(y_val, preds)
            results[name].append(srcc)

    print(f"\n{'─'*50}")
    print(f"Target: {target_name}")
    print(f"{'─'*50}")
    print(f"{'Model':<16} {'Fold SRCCs':<36} {'Mean':>6}")
    print(f"{'─'*50}")

    mean_scores = {}
    for name, scores in results.items():
        scores_str = "  ".join(f"{s:+.3f}" for s in scores)
        mean = np.mean(scores)
        mean_scores[name] = mean
        print(f"{name:<16} {scores_str:<36} {mean:>+.3f}")

    best_name = max(mean_scores, key=mean_scores.get)
    print(f"\n  ✓ Best for {target_name}: {best_name} (SRCC = {mean_scores[best_name]:+.3f})")

    return mean_scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load data
    clip_df = pd.read_csv(CLIP_CSV)
    meta_df = pd.read_csv(METADATA_CSV)

    df = clip_df.merge(meta_df[["id", "channelName", TARGET_VIDEO, TARGET_BRAND]],
                       on="id", how="inner")

    print(f"Loaded {len(df)} videos after merge.")
    print(f"CLIP features: {CLIP_FEATURES}")

    # Check for missing values
    missing = df[CLIP_FEATURES].isnull().sum().sum()
    if missing > 0:
        print(f"[WARN] {missing} missing values in CLIP features — filling with column mean.")
        df[CLIP_FEATURES] = df[CLIP_FEATURES].fillna(df[CLIP_FEATURES].mean())

    X      = df[CLIP_FEATURES].values
    y_vid  = df[TARGET_VIDEO].values
    y_brd  = df[TARGET_BRAND].values
    groups = df["channelName"].values   # GroupKFold key — prevents channel leakage

    # Run CV for both targets
    vid_scores = run_cv(X, y_vid, groups, TARGET_VIDEO)
    brd_scores = run_cv(X, y_brd, groups, TARGET_BRAND)

    # Summary table
    print(f"\n{'═'*50}")
    print("SUMMARY — Mean SRCC")
    print(f"{'═'*50}")
    print(f"{'Model':<16} {TARGET_VIDEO:>22} {TARGET_BRAND:>16}")
    print(f"{'─'*50}")
    for name in vid_scores:
        print(f"{name:<16} {vid_scores[name]:>22.3f} {brd_scores[name]:>16.3f}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
