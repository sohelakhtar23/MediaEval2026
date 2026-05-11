"""
VIDEM Memorability — Test Set Predictor
========================================
Loads artefacts saved by memorability_pipeline.py and generates predictions
for the test set. The pipeline saves TWO separate models:

  mdl_mem   — trained on the best feature set for video memorability
              (all features, optionally with brand_in_title dropped)
  mdl_brand — trained on CNN + LLM features only
              (frame and brand/text features hurt brand prediction in ablation)

Usage
-----
    python predict_test.py \
        --test_csv      testset_videolist_.csv \
        --test_stt      test-stt \
        --test_frames   test-frames \
        --test_features test-features \
        --output        predictions.csv

All --test_* args default to devset paths so you can sanity-check on
training data first (overfit SRCC ≈ 1.0 confirms artefacts are correct).
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import spearmanr, rankdata

# ── Local — shared feature functions from memorability_pipeline ───────────────
from mem_cnn_video_wise_v3_1 import (
    extract_handcrafted_features,
    load_cnn_features,
    compute_tfidf_lsa_features,
    _to_array,
    CACHE_FILE,
    log,
)

# ─────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def rank_normalise(preds: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """
    Map predictions onto the empirical training score distribution using
    rank-based normalisation. Preserves rank order (SRCC unchanged) while
    correcting the RF tendency to compress predictions toward the mean.

    Steps: rank predictions → map to quantiles → index into sorted ref_scores.
    """
    n          = len(preds)
    ranks      = rankdata(preds) - 1
    quantiles  = ranks / (n - 1)
    ref_sorted = np.sort(ref_scores)
    indices    = np.clip((quantiles * (n - 1)).astype(int), 0, n - 1)
    return ref_sorted[indices]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict memorability for test videos")
    p.add_argument("--test_csv",      default="predict/testset_videolist_.csv",
                   help="Test CSV (same columns as devset, no score columns needed)")
    p.add_argument("--test_stt",      default="predict/testset-stt",
                   help="Folder containing {id}.txt STT files for test videos")
    p.add_argument("--test_frames",   default="frames",
                   help="Folder containing frames/{id}/*.jpg for test videos")
    p.add_argument("--test_features", default="predict/testset-features",
                   help="Folder containing features/AlexNet/{id}.npy etc.")
    p.add_argument("--artefacts",     default="model_artefacts.pkl",
                   help="Pickle saved by memorability_pipeline.py")
    p.add_argument("--output",        default="predict/predictions.csv",
                   help="Output CSV path")
    p.add_argument("--cache",         default="predict/llm_scalar_cache_test.json",
                   help="Separate LLM cache for test set")
    return p.parse_args()


def load_artefacts(path: str) -> dict:
    with open(path, "rb") as fh:
        art = pickle.load(fh)
    log.info(f"Artefacts loaded from {path}")
    log.info(f"  Mem   model config : {art.get('best_mem_config',   'all features')}")
    log.info(f"  Brand model config : {art.get('best_brand_config', 'cnn+llm only')}")
    log.info(f"  Mem   feature dim  : {len(art['mem_feature_names'])}")
    log.info(f"  Brand feature dim  : {len(art['brand_feature_names'])}")
    return art


def assemble_mem_features(X_frame: np.ndarray,
                           X_brand_feat: np.ndarray,
                           X_llm: np.ndarray,
                           X_cnn: np.ndarray,
                           art: dict,
                           brand_col_names: list) -> np.ndarray:
    """
    Reconstruct the mem feature matrix exactly as it was during training.
    If brand_in_title was dropped (mem_drop_bit=True), removes it here too.
    """
    full_X = np.concatenate([X_frame, X_brand_feat, X_llm, X_cnn], axis=1)
    if art.get("mem_drop_bit", False):
        bit_idx  = brand_col_names.index("brand_in_title")
        drop_col = X_frame.shape[1] + bit_idx
        mask     = np.ones(full_X.shape[1], dtype=bool)
        mask[drop_col] = False
        full_X   = full_X[:, mask]
        log.info("  brand_in_title dropped from mem features (as in training)")
    return full_X


def assemble_brand_features(X_cnn: np.ndarray,
                             X_lsa: np.ndarray,
                             best_brand_config: str) -> np.ndarray:
    """
    Assemble the brand feature matrix matching the config selected during training:
      'B: LSA-only (brand)'  → LSA features only
      'C: CNN+LSA (brand)'   → CNN + LSA concatenated
      anything else          → CNN only (fallback)
    """
    if best_brand_config == "B: LSA-only (brand)":
        return X_lsa
    elif best_brand_config == "C: CNN+LSA (brand)":
        return np.concatenate([X_cnn, X_lsa], axis=1)
    else:
        return X_cnn


def predict(X_mem: np.ndarray, X_brand: np.ndarray, art: dict) -> tuple:
    """
    Scale → predict → rank-normalise onto training score distribution.

    Rank-normalisation preserves the prediction ranking (SRCC unchanged)
    while mapping the output onto the empirical training distribution,
    correcting the RF tendency to compress predictions toward the mean.
    """
    pred_mem   = art["mdl_mem"].predict(
                     art["mem_scaler"].transform(X_mem))
    pred_brand = art["mdl_brand"].predict(
                     art["brand_scaler"].transform(X_brand))

    # Rank-normalise using training score distributions
    pred_mem   = rank_normalise(pred_mem,   art["y_mem_train"]).clip(0, 1)
    pred_brand = rank_normalise(pred_brand, art["y_brand_train"]).clip(0, 1)

    return pred_mem, pred_brand


def main():
    args = parse_args()

    test_csv      = Path(args.test_csv)
    test_stt      = Path(args.test_stt)
    test_frames   = Path(args.test_frames)
    test_features = Path(args.test_features)
    cache_path    = Path(args.cache)
    output_path   = Path(args.output)

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    # ── Load artefacts ─────────────────────────────────────────────────────────
    art = load_artefacts(args.artefacts)

    n_cnn = art["n_cnn_features"]     # how many PCA dims the training used

    # ── Load test CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(test_csv)
    log.info(f"Test set: {len(df)} videos  |  "
             f"channels: {df['channelName'].nunique()}")

    # ── OpenAI client (uses same cache logic as training) ─────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None
    if not openai_client:
        log.warning("OPENAI_API_KEY not set — LLM scalars will be neutral (5.0)")

    cache: dict = {}
    if cache_path.exists():
        with open(cache_path) as fh:
            cache = json.load(fh)
        log.info(f"LLM cache (test): {len(cache)} entries loaded")

    # ── Extract handcrafted features (dirs 1, 2, 3) ───────────────────────────
    log.info("Extracting features for test set ...")
    frame_df, brand_df, llm_df = extract_handcrafted_features(
        df, test_stt, test_frames, cache, openai_client)

    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=2)
    log.info(f"LLM cache (test) saved: {len(cache)} entries")

    # ── Direction 4: CNN features — inference mode ────────────────────────────
    log.info("Loading CNN features for test set ...")
    X_cnn, _, _ = load_cnn_features(
        df,
        features_root=test_features,
        pca_model=art["pca_model"],
        scaler_model=art["cnn_scaler"],
    )

    # ── Direction 6: TF-IDF + LSA — inference mode (reuse train models) ───────
    log.info("Computing TF-IDF + LSA features for test set ...")
    X_lsa, _, _ = compute_tfidf_lsa_features(
        df, test_stt,
        tfidf_model=art["tfidf_model"],
        lsa_model=art["lsa_model"],
    )

    # Verify CNN dims match what was saved
    if X_cnn.shape[1] != n_cnn:
        raise ValueError(
            f"CNN feature dim mismatch: got {X_cnn.shape[1]}, "
            f"expected {n_cnn}. "
            f"Ensure the same CNN model folders exist in test_features."
        )

    X_frame      = _to_array(frame_df)
    X_brand_feat = _to_array(brand_df)
    X_llm        = _to_array(llm_df)

    # ── Assemble feature matrices ──────────────────────────────────────────────
    X_mem_test   = assemble_mem_features(
        X_frame, X_brand_feat, X_llm, X_cnn,
        art, list(brand_df.columns)
    )
    X_brand_test = assemble_brand_features(
        X_cnn, X_lsa, art["best_brand_config"]
    )

    # Final dimension check
    n_mem_expected   = len(art["mem_feature_names"])
    n_brand_expected = len(art["brand_feature_names"])

    if X_mem_test.shape[1] != n_mem_expected:
        raise ValueError(
            f"Mem feature dim mismatch: got {X_mem_test.shape[1]}, "
            f"expected {n_mem_expected}"
        )
    if X_brand_test.shape[1] != n_brand_expected:
        raise ValueError(
            f"Brand feature dim mismatch: got {X_brand_test.shape[1]}, "
            f"expected {n_brand_expected}"
        )

    log.info(f"Test mem   features: {X_mem_test.shape}")
    log.info(f"Test brand features: {X_brand_test.shape}")

    # ── Predict ────────────────────────────────────────────────────────────────
    pred_mem, pred_brand = predict(X_mem_test, X_brand_test, art)

    log.info(f"mem   pred — mean={pred_mem.mean():.4f}  "
             f"std={pred_mem.std():.4f}  "
             f"range=[{pred_mem.min():.3f}, {pred_mem.max():.3f}]")
    log.info(f"brand pred — mean={pred_brand.mean():.4f}  "
             f"std={pred_brand.std():.4f}  "
             f"range=[{pred_brand.min():.3f}, {pred_brand.max():.3f}]")

    # ── Write output CSV ───────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "video_id":            df["video_id"].values,
        "id":                  df["id"].values,
        "memorability_score":  np.round(pred_mem,   6),
        "brand_memorability":  np.round(pred_brand, 6),
    })
    out_df.to_csv(output_path, index=False)
    log.info(f"Predictions saved → {output_path.resolve()}  ({len(out_df)} rows)")

    # ── Sanity check if ground truth accidentally present ─────────────────────
    if "memorability_score" in df.columns and "brand_memorability" in df.columns:
        srcc_m = spearmanr(df["memorability_score"], pred_mem).correlation
        srcc_b = spearmanr(df["brand_memorability"],  pred_brand).correlation
        log.info(f"Ground truth found — SRCC mem={srcc_m:.4f}  brand={srcc_b:.4f}")
        if srcc_m > 0.9:
            log.info("(High SRCC = sanity-check on train data passed ✓)")

    # ── Print preview of first 10 predictions ─────────────────────────────────
    print("\n  First 10 predictions:")
    print(f"  {'id':<15}  {'channel':<22}  {'mem':>6}  {'brand':>6}")
    print(f"  {'-'*15}  {'-'*22}  {'-'*6}  {'-'*6}")
    for _, row in out_df.head(10).iterrows():
        ch = df.loc[df["id"] == row["id"], "channelName"].values
        ch = ch[0][:22] if len(ch) else "?"
        print(f"  {row['id']:<15}  {ch:<22}  "
              f"{row['memorability_score']:>6.4f}  "
              f"{row['brand_memorability']:>6.4f}")

    print(f"\n{'✓ Predictions complete.':^64}\n")
    print(f"  Output: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()