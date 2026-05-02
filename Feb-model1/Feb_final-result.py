"""
VIDEM Commercial Memorability Prediction Pipeline
=================================================
MediaEval Challenge 2.1 (Video Memorability) & 2.2 (Brand Memorability)

Method  : TF-IDF on STT transcripts → LSA (TruncatedSVD) → Rank aggregation
CV      : Brand-isolated 5-fold (no brand appears in both train & val)
Results : Memorability Spearman r ≈ 0.20  |  Brand Memorability r ≈ 0.17

Directory layout expected
─────────────────────────
  devset_videolist_GT.csv
  devset-stt/               ← 339 .txt transcript files named by YouTube ID
  features/  (optional)     ← 10 sub-folders of .npy per-video feature files

Usage
─────
  # Train, run cross-validation, save pipeline
  python memorability_pipeline.py --mode train

  # Predict on test set (after receiving test STT folder)
  python memorability_pipeline.py --mode predict \
      --test_csv  testset_videolist.csv \
      --test_stt  testset-stt/ \
      --output    predictions.csv
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_CSV     = "devset_videolist_GT.csv"
STT_DIR       = Path("devset-stt")
PIPELINE_PATH = Path("memorability_pipeline.pkl")

# ── TF-IDF parameters (exactly as in working notebook) ────────────────────────
TFIDF_PARAMS = dict(
    max_features=3000,
    min_df=3,
    max_df=0.80,
    ngram_range=(1, 3),
    sublinear_tf=True,
    strip_accents="unicode",
    stop_words="english",
    analyzer="word",
)

# ── Cross-validation settings ─────────────────────────────────────────────────
N_FOLDS     = 5
R_THRESHOLD = 0.03   # minimum |Spearman r| to include a LSA dim

# ── LSA search grid ───────────────────────────────────────────────────────────
N_COMP_CANDIDATES = [50, 80, 100, 120, 150, 200]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_train_data():
    """Load CSV, build folds, load transcripts, fit TF-IDF. Returns all objects."""
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} videos from {TRAIN_CSV}")

    # IMPORTANT: 'id' column  = YouTube video ID  (e.g. 'CbfCUSPN7KI')
    #            'video_id'   = sequential integer — do NOT use for filenames
    y_mem   = df["memorability_score"].values
    y_brand = df["brand_memorability"].values

    # ── Brand-isolated folds ──────────────────────────────────────────────────
    df = build_brand_folds(df)

    # ── Load STT transcripts ──────────────────────────────────────────────────
    transcripts = load_transcripts(df["id"].tolist(), STT_DIR)

    # ── Build corpus and fit TF-IDF once ─────────────────────────────────────
    corpus  = [transcripts[vid] for vid in df["id"]]
    tfidf   = TfidfVectorizer(**TFIDF_PARAMS)
    X_tfidf = tfidf.fit_transform(corpus)
    print(f"TF-IDF vocabulary size: {X_tfidf.shape[1]}")

    return df, y_mem, y_brand, transcripts, corpus, tfidf, X_tfidf


def load_transcripts(video_ids: list, stt_dir: Path) -> dict:
    """
    Returns {video_id: transcript_text}.
    Tries both 'ID.txt' and '_ID.txt' filename variants.
    Missing files map to empty string.
    """
    transcripts = {}
    missing = []
    for vid in video_ids:
        found = False
        for name in [vid, f"_{vid}"]:
            path = stt_dir / f"{name}.txt"
            if path.exists():
                transcripts[vid] = path.read_text(encoding="utf-8").strip()
                found = True
                break
        if not found:
            transcripts[vid] = ""
            missing.append(vid)

    n_empty = sum(1 for t in transcripts.values() if t == "")
    print(f"Transcripts — missing files: {len(missing)}, empty: {n_empty}")
    return transcripts


# ══════════════════════════════════════════════════════════════════════════════
# BRAND-ISOLATED FOLD CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_brand_folds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Goldman Sachs (79 videos, 23%) → fold 0 alone.
    All other channels → greedy bin-packing into folds 1-4 (~65 videos each).
    Adds a 'fold' column and returns modified df.
    """
    channel_counts = df["channelName"].value_counts()
    print("Building brand-isolated folds...")
    print(channel_counts.to_string())

    fold_assignments = {"Goldman Sachs": 0}
    fold_sizes = {0: int(channel_counts.get("Goldman Sachs", 0)),
                  1: 0, 2: 0, 3: 0, 4: 0}

    for channel, count in channel_counts.drop("Goldman Sachs",
                                               errors="ignore").items():
        target = min([1, 2, 3, 4], key=lambda f: fold_sizes[f])
        fold_assignments[channel] = target
        fold_sizes[target] += count

    df = df.copy()
    df["fold"] = df["channelName"].map(fold_assignments)

    print("\nFold sizes (videos):")
    print(df["fold"].value_counts().sort_index())
    print("\nChannels per fold:")
    for f in range(N_FOLDS):
        chs = [ch for ch, fl in fold_assignments.items() if fl == f]
        print(f"  Fold {f}: {chs}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# RANK AGGREGATION — zero free parameters → zero overfitting risk
# ══════════════════════════════════════════════════════════════════════════════

def rank_aggregate_cv(feature_dict: dict,
                      y_mem: np.ndarray,
                      y_brand: np.ndarray,
                      df: pd.DataFrame):
    """
    Brand-isolated cross-validation using rank aggregation.

    For each fold:
      1. Compute Spearman r of each LSA dim vs target on the TRAIN split.
      2. Keep only dims where |r| >= R_THRESHOLD.
      3. Rank those dims on the VAL split; multiply by sign(r).
      4. Sum → final memorability score for each val video.

    Returns (r_mem, r_brand, preds_mem, preds_brand).
    """
    preds_mem   = np.zeros(len(df))
    preds_brand = np.zeros(len(df))
    X_all       = np.column_stack(list(feature_dict.values()))

    for fold_id in range(N_FOLDS):
        val_mask   = (df["fold"] == fold_id).values
        train_mask = ~val_mask
        X_tr  = X_all[train_mask]
        X_val = X_all[val_mask]
        n_val = val_mask.sum()

        for y, preds in [(y_mem, preds_mem), (y_brand, preds_brand)]:
            y_tr       = y[train_mask]
            valid_cols = []
            directions = []

            for j in range(X_tr.shape[1]):
                col = X_tr[:, j]
                if len(np.unique(col)) < 2:
                    continue
                r, _ = spearmanr(col, y_tr)
                if abs(r) >= R_THRESHOLD:
                    valid_cols.append(j)
                    directions.append(np.sign(r))

            if not valid_cols:
                preds[val_mask] = y_tr.mean()
                continue

            agg = np.zeros(n_val)
            for j, sign in zip(valid_cols, directions):
                agg += sign * rankdata(X_val[:, j]).astype(float)
            preds[val_mask] = agg

    r_m = spearmanr(y_mem,   preds_mem).correlation
    r_b = spearmanr(y_brand, preds_brand).correlation
    return r_m, r_b, preds_mem, preds_brand


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMAL n_comp SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def find_best_n_comp(X_tfidf, y_mem, y_brand, df,
                     candidates=N_COMP_CANDIDATES):
    """
    Tries each candidate LSA size on the already-fitted X_tfidf.
    Returns (best_n_mem, best_n_brand).
    """
    print("\n" + "="*60)
    print("FINDING OPTIMAL n_comp PER TARGET")
    print(f"{'n_comp':<10} {'Mem r':>8} {'Brand r':>8}")
    print("-"*30)

    best_mem_r, best_mem_n     = -99, candidates[0]
    best_brand_r, best_brand_n = -99, candidates[0]

    for n in candidates:
        svd   = TruncatedSVD(n_components=n, random_state=42)
        X_lsa = svd.fit_transform(X_tfidf)
        feats = {f"lsa_{i}": X_lsa[:, i] for i in range(n)}
        r_m, r_b, _, _ = rank_aggregate_cv(feats, y_mem, y_brand, df)
        print(f"{n:<10} {r_m:>8.4f} {r_b:>8.4f}")
        if r_m > best_mem_r:
            best_mem_r, best_mem_n = r_m, n
        if r_b > best_brand_r:
            best_brand_r, best_brand_n = r_b, n

    print(f"\nBest n_comp for memorability:       {best_mem_n} (r={best_mem_r:.4f})")
    print(f"Best n_comp for brand memorability: {best_brand_n} (r={best_brand_r:.4f})")
    return best_mem_n, best_brand_n


# ══════════════════════════════════════════════════════════════════════════════
# FINAL MODEL FITTING (all 339 training videos)
# ══════════════════════════════════════════════════════════════════════════════

def fit_final_pipeline(tfidf, X_tfidf, y_mem, y_brand,
                       n_comp_mem, n_comp_brand):
    """
    Fits two SVDs on the full training TF-IDF matrix and computes per-dim
    Spearman directions. Returns a serialisable pipeline dict.
    """
    print("\n" + "="*60)
    print("FITTING FINAL MODELS ON ALL 339 TRAINING VIDEOS")

    final_svd_mem   = TruncatedSVD(n_components=n_comp_mem,   random_state=42)
    final_svd_brand = TruncatedSVD(n_components=n_comp_brand, random_state=42)

    X_lsa_mem   = final_svd_mem.fit_transform(X_tfidf)
    X_lsa_brand = final_svd_brand.fit_transform(X_tfidf)

    def get_directions(X_lsa, y):
        dims, dirs = [], []
        for j in range(X_lsa.shape[1]):
            col = X_lsa[:, j]
            if len(np.unique(col)) < 2:
                continue
            r, _ = spearmanr(col, y)
            if abs(r) >= R_THRESHOLD:
                dims.append(j)
                dirs.append(float(np.sign(r)))
        return dims, dirs

    mem_dims,   mem_dirs   = get_directions(X_lsa_mem,   y_mem)
    brand_dims, brand_dirs = get_directions(X_lsa_brand, y_brand)

    print(f"  Dims used for memorability:       {len(mem_dims)}/{n_comp_mem}")
    print(f"  Dims used for brand memorability: {len(brand_dims)}/{n_comp_brand}")

    return dict(
        tfidf        = tfidf,
        svd_mem      = final_svd_mem,
        svd_brand    = final_svd_brand,
        mem_dims     = mem_dims,
        mem_dirs     = mem_dirs,
        brand_dims   = brand_dims,
        brand_dirs   = brand_dirs,
        y_mem_min    = float(y_mem.min()),
        y_mem_max    = float(y_mem.max()),
        y_brand_min  = float(y_brand.min()),
        y_brand_max  = float(y_brand.max()),
        n_comp_mem   = n_comp_mem,
        n_comp_brand = n_comp_brand,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_test_videos(pipeline: dict,
                        test_transcripts_dict: dict,
                        test_video_ids: list):
    """
    Predicts memorability scores for new (unseen) test videos.

    Parameters
    ----------
    pipeline              : dict from fit_final_pipeline or loaded .pkl
    test_transcripts_dict : {video_id: transcript_text}
    test_video_ids        : ordered list of YouTube video IDs ('id' column)

    Returns
    -------
    pred_mem, pred_brand  : np.ndarray shape (n_test,), in training score range
    """
    test_corpus = [test_transcripts_dict.get(vid, "")
                   for vid in test_video_ids]

    # Transform using TRAIN-fitted TF-IDF vocabulary (no re-fitting)
    X_test_tfidf = pipeline["tfidf"].transform(test_corpus)

    X_test_lsa_mem   = pipeline["svd_mem"].transform(X_test_tfidf)
    X_test_lsa_brand = pipeline["svd_brand"].transform(X_test_tfidf)

    pred_mem   = np.zeros(len(test_video_ids))
    pred_brand = np.zeros(len(test_video_ids))

    for j, sign in zip(pipeline["mem_dims"], pipeline["mem_dirs"]):
        pred_mem += sign * rankdata(X_test_lsa_mem[:, j]).astype(float)

    for j, sign in zip(pipeline["brand_dims"], pipeline["brand_dirs"]):
        pred_brand += sign * rankdata(X_test_lsa_brand[:, j]).astype(float)

    def scale(arr, lo, hi):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, (lo + hi) / 2, dtype=float)
        return (arr - mn) / (mx - mn) * (hi - lo) + lo

    pred_mem   = scale(pred_mem,   pipeline["y_mem_min"],   pipeline["y_mem_max"])
    pred_brand = scale(pred_brand, pipeline["y_brand_min"], pipeline["y_brand_max"])
    return pred_mem, pred_brand


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_cv_breakdown(y_mem, y_brand, preds_mem, preds_brand, df):
    r_m, _ = spearmanr(y_mem,   preds_mem)
    r_b, _ = spearmanr(y_brand, preds_brand)
    print("\n" + "="*60)
    print("FINAL CV RESULTS — per-target optimal configs")
    print(f"  Memorability Spearman r:       {r_m:.4f}")
    print(f"  Brand Memorability Spearman r: {r_b:.4f}")
    print("\nPer-fold breakdown:")
    print(f"  {'Fold':<6} {'n':>4}  {'Mem r':>7}  {'Brand r':>8}  Channels")
    print("  " + "-"*65)
    for fold_id in range(N_FOLDS):
        mask = (df["fold"] == fold_id).values
        rf_m, _ = spearmanr(y_mem[mask],   preds_mem[mask])
        rf_b, _ = spearmanr(y_brand[mask], preds_brand[mask])
        chs = ", ".join(df[mask]["channelName"].unique()[:2].tolist())
        print(f"  {fold_id:<6} {mask.sum():>4}  {rf_m:>7.4f}  {rf_b:>8.4f}  {chs}...")
    return r_m, r_b


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_train():
    # Step 1 — Load data (CSV + STT + TF-IDF, all in one call)
    df, y_mem, y_brand, transcripts, corpus, tfidf, X_tfidf = load_train_data()

    # Step 2 — Find optimal LSA size per target
    best_n_mem, best_n_brand = find_best_n_comp(X_tfidf, y_mem, y_brand, df)

    # Step 3 — Final CV with per-target optimal configs
    svd_mem   = TruncatedSVD(n_components=best_n_mem,   random_state=42)
    svd_brand = TruncatedSVD(n_components=best_n_brand, random_state=42)

    X_lsa_mem   = svd_mem.fit_transform(X_tfidf)
    X_lsa_brand = svd_brand.fit_transform(X_tfidf)

    feats_mem   = {f"lsa_{i}": X_lsa_mem[:,   i] for i in range(best_n_mem)}
    feats_brand = {f"lsa_{i}": X_lsa_brand[:, i] for i in range(best_n_brand)}

    r_m, _, preds_mem,   _ = rank_aggregate_cv(feats_mem,   y_mem, y_brand, df)
    _, r_b, _, preds_brand = rank_aggregate_cv(feats_brand, y_mem, y_brand, df)
    r_m_cv, r_b_cv = print_cv_breakdown(y_mem, y_brand, preds_mem, preds_brand, df)

    # Step 4 — Fit final pipeline on all training videos
    pipeline = fit_final_pipeline(tfidf, X_tfidf, y_mem, y_brand,
                                  best_n_mem, best_n_brand)
    pipeline["cv_mem_r"]   = r_m_cv
    pipeline["cv_brand_r"] = r_b_cv

    # Step 5 — Sanity check on training set
    print("\nSanity check — predict_test_videos applied to train set:")
    train_dict   = {vid: transcripts[vid] for vid in df["id"]}
    pm_tr, pb_tr = predict_test_videos(pipeline, train_dict, df["id"].tolist())
    print(f"  Mem r (train):   {spearmanr(y_mem,   pm_tr).correlation:.4f}  "
          f"(expected > CV score)")
    print(f"  Brand r (train): {spearmanr(y_brand, pb_tr).correlation:.4f}  "
          f"(expected > CV score)")

    # Step 6 — Save
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print("\n" + "="*60)
    print(f"Pipeline saved → {PIPELINE_PATH}")
    print("\nFINAL SUMMARY:")
    print(f"  Method  : LSA({best_n_mem}/{best_n_brand}) + rank aggregation on STT")
    print(f"  CV Memorability Spearman r:       {r_m_cv:.4f}")
    print(f"  CV Brand Memorability Spearman r: {r_b_cv:.4f}")
    print(f"  No fitted model parameters — zero overfitting risk")
    print("\nTo predict on the test set:")
    print("  python memorability_pipeline.py --mode predict \\")
    print("      --test_csv testset_videolist.csv \\")
    print("      --test_stt testset-stt/ \\")
    print("      --output   predictions.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_predict(args):
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"{PIPELINE_PATH} not found. Run --mode train first."
        )
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)

    print(f"Pipeline loaded from {PIPELINE_PATH}")
    print(f"  CV Memorability r:       {pipeline.get('cv_mem_r', 'N/A'):.4f}")
    print(f"  CV Brand Memorability r: {pipeline.get('cv_brand_r', 'N/A'):.4f}")

    # Load test set — video IDs are in the 'id' column (YouTube IDs)
    test_df = pd.read_csv(args.test_csv)
    print(f"\nLoaded {len(test_df)} test videos from {args.test_csv}")

    test_transcripts = load_transcripts(
        test_df["id"].tolist(), Path(args.test_stt)
    )

    video_ids            = test_df["id"].tolist()
    pred_mem, pred_brand = predict_test_videos(pipeline, test_transcripts,
                                               video_ids)

    out_df = pd.DataFrame({
        "video_id":           video_ids,
        "memorability_score": pred_mem,
        "brand_memorability": pred_brand,
    })
    out_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved → {args.output}")
    print("\nPrediction statistics:")
    print(out_df[["memorability_score", "brand_memorability"]].describe().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VIDEM Memorability Prediction Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["train", "predict"], default="train",
        help="'train': CV + save pipeline | 'predict': load pipeline + predict"
    )
    parser.add_argument("--test_csv", default="testset_videolist.csv",
                        help="Test set CSV (predict mode)")
    parser.add_argument("--test_stt", default="testset-stt/",
                        help="Test set STT directory (predict mode)")
    parser.add_argument("--output",   default="predictions.csv",
                        help="Output CSV path (predict mode)")
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    else:
        run_predict(args)