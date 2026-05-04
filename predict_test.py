"""
VIDEM Memorability — Test Set Predictor
========================================
Loads artefacts saved by memorability_pipeline.py, extracts the same
features for the test set, and writes predictions to a CSV.

Usage
-----
    python predict_test.py \
        --test_csv   path/to/testset_videolist.csv \
        --test_stt   path/to/test-stt \
        --test_frames path/to/test-frames \
        --test_features path/to/test-features \
        --output     predictions.csv

All --test_* arguments default to the devset paths so you can sanity-check
on training data first (should overfit, SRCC ≈ 1.0 if correct).

The output CSV format matches the MediaEval submission format:
    video_id, memorability_score, brand_memorability
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
from PIL import Image

# ── Local — shared feature functions from memorability_pipeline ───────────────
from memorability_pipeline import (
    compute_frame_features,
    compute_brand_features,
    get_llm_scalars,
    load_cnn_features,
    extract_handcrafted_features,
    _to_array,
    CACHE_FILE,
    OPENAI_MODEL,
    log,
)

# ─────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
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
    log.info(f"  Feature names ({len(art['feature_names'])}): "
             f"{art['feature_names'][:5]} ...")
    return art


def predict(X: np.ndarray, art: dict) -> tuple:
    """
    Apply final_scaler → mdl_mem / mdl_brand and return (pred_mem, pred_brand).
    Clips predictions to [0, 1] since memorability scores are bounded.
    """
    X_scaled    = art["final_scaler"].transform(X)
    pred_mem    = art["mdl_mem"].predict(X_scaled).clip(0, 1)
    pred_brand  = art["mdl_brand"].predict(X_scaled).clip(0, 1)
    return pred_mem, pred_brand


def main():
    args = parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────────
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

    # ── Load test CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(test_csv)
    log.info(f"Test set: {len(df)} videos")

    # ── OpenAI client (uses same API key; test LLM calls cached separately) ────
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None
    if not openai_client:
        log.warning("OPENAI_API_KEY not set — LLM scalars will be neutral (5.0)")

    cache: dict = {}
    if cache_path.exists():
        with open(cache_path) as fh:
            cache = json.load(fh)
        log.info(f"LLM cache (test): {len(cache)} entries")

    # ── Extract handcrafted features (dirs 1, 2, 3) ───────────────────────────
    log.info("Extracting features for test set ...")
    frame_df, brand_df, llm_df = extract_handcrafted_features(
        df, test_stt, test_frames, cache, openai_client, is_training=False)

    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=2)
    log.info(f"LLM cache (test) saved ({len(cache)} entries)")

    # ── Direction 4: CNN features (inference mode — use saved PCA/scaler) ──────
    log.info("Loading CNN features for test set ...")
    X_cnn, _, _ = load_cnn_features(
        df,
        features_root=test_features,
        pca_model=art["pca_model"],
        scaler_model=art["cnn_scaler"],
    )

    X_frame = _to_array(frame_df)
    X_brand = _to_array(brand_df)
    X_llm   = _to_array(llm_df)

    # Stack in the same order as training
    parts = [X_frame, X_brand, X_llm]
    if X_cnn.shape[1] > 0:
        parts.append(X_cnn)
    full_X = np.concatenate(parts, axis=1)

    # Verify dimensionality matches training
    n_train_features = len(art["feature_names"])
    if full_X.shape[1] != n_train_features:
        raise ValueError(
            f"Feature dimension mismatch: got {full_X.shape[1]}, "
            f"expected {n_train_features}. "
            f"Check that the same CNN models are present in test_features."
        )
    log.info(f"Test feature matrix: {full_X.shape}")

    # ── Predict ────────────────────────────────────────────────────────────────
    pred_mem, pred_brand = predict(full_X, art)

    log.info(f"Predictions: mem   mean={pred_mem.mean():.4f}  "
             f"std={pred_mem.std():.4f}  "
             f"range=[{pred_mem.min():.4f}, {pred_mem.max():.4f}]")
    log.info(f"             brand mean={pred_brand.mean():.4f}  "
             f"std={pred_brand.std():.4f}  "
             f"range=[{pred_brand.min():.4f}, {pred_brand.max():.4f}]")

    # ── Write output CSV ───────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "video_id":          df["video_id"].values,
        "id":                df["id"].values,
        "memorability_score":   pred_mem,
        "brand_memorability":   pred_brand,
    })
    out_df.to_csv(output_path, index=False)
    log.info(f"Predictions saved to {output_path} ({len(out_df)} rows)")

    # ── If ground truth is available, report SRCC ──────────────────────────────
    if "memorability_score" in df.columns and "brand_memorability" in df.columns:
        from scipy.stats import spearmanr
        srcc_m = spearmanr(df["memorability_score"], pred_mem).correlation
        srcc_b = spearmanr(df["brand_memorability"],  pred_brand).correlation
        log.info(f"Ground truth available — SRCC mem={srcc_m:.4f}  brand={srcc_b:.4f}")
        log.info("(This is train-set SRCC — should be ~1.0 if artefacts are correct)")

    print(f"\n{'✓ Predictions complete.':^64}\n")
    print(f"  Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
