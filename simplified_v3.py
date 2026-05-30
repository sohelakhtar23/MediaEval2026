"""
VIDEM Memorability Prediction Pipeline
Multi-stream RF regression for video and brand memorability (339 videos, 32 channels).

Feature streams:
  1. Frame features     — 14 visual scalars from first 60 frames
  2. Brand/text         — 8 string-matching features from STT + metadata
  3. LLM scalars        — 8 semantic scores loaded from pre-generated JSON cache
  4. CNN features       — 7 architectures, PCA-32
"""

import json
import logging
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path(".")
STT_DIR      = DATA_ROOT / "devset-stt"
FRAMES_DIR   = DATA_ROOT / "frames"
FEATURES_DIR = DATA_ROOT / "features"
CACHE_FILE   = DATA_ROOT / "llm_scalar_cache.json"

CNN_MODELS  = ["AlexNet", "ResNet50", "VGG", "ViT", "EfficientNetB3", "DenseNet121", "R3D"]
PCA_DIM     = 32
N_FOLDS     = 5
RANDOM_SEED = 42

_LLM_KEYS = [
    "brand_prominence", "emotional_valence", "narrative_arc", "call_to_action",
    "information_density", "novelty_surprise", "visual_dynamism", "brand_specificity",
]

_FRAME_KEYS = [
    "frame_r_mean", "frame_g_mean", "frame_b_mean",
    "frame_r_std",  "frame_g_std",  "frame_b_std",
    "frame_brightness", "frame_bright_var",
    "frame_saturation", "frame_sat_var",
    "frame_motion",     "frame_motion_var",
    "frame_n_cuts",     "frame_color_change",
]


# ── Direction 1: First-minute frame features ──────────────────────────────────

def compute_frame_features(video_id: str) -> dict:
    """14 visual scalars from the first 60 frames (1fps = first minute the annotator watched)."""
    frame_dir = FRAMES_DIR / video_id
    frames = sorted(frame_dir.iterdir())[:60] if frame_dir.exists() else []
    if not frames:
        return {k: 0.0 for k in _FRAME_KEYS}

    pixel_means, pixel_stds, brightness_vals, sat_vals, diffs = [], [], [], [], []
    prev_arr = None

    for fp in frames:
        try:
            arr = np.array(Image.open(fp).convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
            pixel_means.append(arr.mean(axis=(0, 1)))
            pixel_stds.append(arr.std(axis=(0, 1)))
            brightness_vals.append(arr.mean())
            sat_vals.append(arr.std(axis=2).mean())
            if prev_arr is not None:
                diffs.append(np.abs(arr - prev_arr).mean())
            prev_arr = arr
        except Exception as e:
            log.debug(f"Skipping {fp.name}: {e}")

    if not pixel_means:
        return {k: 0.0 for k in _FRAME_KEYS}

    pm = np.stack(pixel_means)
    ps = np.stack(pixel_stds)
    return {
        "frame_r_mean":       float(pm[:, 0].mean()),
        "frame_g_mean":       float(pm[:, 1].mean()),
        "frame_b_mean":       float(pm[:, 2].mean()),
        "frame_r_std":        float(ps[:, 0].mean()),
        "frame_g_std":        float(ps[:, 1].mean()),
        "frame_b_std":        float(ps[:, 2].mean()),
        "frame_brightness":   float(np.mean(brightness_vals)),
        "frame_bright_var":   float(np.std(brightness_vals)),
        "frame_saturation":   float(np.mean(sat_vals)),
        "frame_sat_var":      float(np.std(sat_vals)),
        "frame_motion":       float(np.mean(diffs)) if diffs else 0.0,
        "frame_motion_var":   float(np.std(diffs))  if diffs else 0.0,
        "frame_n_cuts":       float(sum(1 for d in diffs if d > 0.05)),
        "frame_color_change": float(pm.std(axis=0).mean()),
    }


# ── Direction 2: Brand entity density ─────────────────────────────────────────

def compute_brand_features(row: pd.Series, stt_text: str) -> dict:
    """8 string-matching brand and text features (no external NLP library)."""
    brand = str(row.get("channelName", "")).lower().strip()
    title = str(row.get("title", "")).lower()
    tags  = str(row.get("tags", "")).lower()
    stt   = stt_text.lower()

    all_text    = " ".join([title, str(row.get("description", "")).lower(), tags, stt])
    stt_words   = stt.split()
    n_stt_words = max(len(stt_words), 1)
    brand_count = all_text.count(brand)
    sentences   = [s.strip() for s in stt.replace("?", ".").replace("!", ".").split(".") if s.strip()]

    return {
        "brand_mention_count": float(brand_count),
        "brand_density":       float(brand_count / max(len(all_text.split()), 1)),
        "brand_in_title":      float(brand in title),
        "brand_in_stt":        float(brand in stt),
        "stt_word_count":      float(len(stt_words)),
        "stt_unique_ratio":    float(len(set(stt_words)) / n_stt_words),
        "avg_sentence_len":    float(n_stt_words / max(len(sentences), 1)),
        "tag_count":           float(len([t for t in tags.split(",") if t.strip()])),
    }


# ── Direction 3: LLM scalars (loaded from pre-generated cache) ────────────────

def load_llm_cache() -> dict:
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    log.info(f"LLM cache loaded: {len(cache)} entries")
    return cache

def get_llm_scalars(video_id: str, cache: dict) -> dict:
    """Return pre-generated LLM scalars from cache; default 5.0 if missing."""
    return cache.get(video_id, {k: 5.0 for k in _LLM_KEYS})


# ── Direction 4: Pre-extracted CNN features ────────────────────────────────────

def compute_cnn_features(vid_id: str) -> np.ndarray | None:
    """Concatenate mean+std across time for each CNN model. Returns None if no files found."""
    parts = []
    for model_name in CNN_MODELS:
        npy = FEATURES_DIR / model_name / f"{vid_id}.npy"
        if not npy.exists():
            continue
        arr = np.load(npy, allow_pickle=True)
        if arr.ndim == 2:
            parts.append(np.concatenate([arr.mean(0), arr.std(0)]))
        elif arr.ndim == 1:
            parts.append(arr)
        else:
            parts.append(arr.flatten())
    return np.concatenate(parts) if parts else None


def apply_cnn_pca(cnn_raw: np.ndarray, pca_model=None, scaler_model=None):
    """Fit (training) or apply (inference) scaler + PCA-32 to the raw CNN matrix."""
    log.info(f"Concatenated CNN shape: {cnn_raw.shape}")
    if scaler_model is None:
        scaler_model = StandardScaler()
        X_scaled = scaler_model.fit_transform(cnn_raw)
    else:
        X_scaled = scaler_model.transform(cnn_raw)

    if pca_model is None:
        pca_model = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
        X_pca = pca_model.fit_transform(X_scaled)
        log.info(f"PCA-{PCA_DIM} variance explained: {pca_model.explained_variance_ratio_.sum():.2%}")
    else:
        X_pca = pca_model.transform(X_scaled)

    return X_pca, pca_model, scaler_model


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame, llm_cache: dict) -> tuple:
    """Run all 4 feature directions for every video. Returns (frame_df, brand_df, llm_df, cnn_raw)."""
    frame_rows, brand_rows, llm_rows, cnn_rows = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        vid_id   = row["id"]
        stt_path = STT_DIR / f"{vid_id}.txt"
        stt_text = stt_path.read_text(encoding="utf-8", errors="ignore") if stt_path.exists() else ""

        frame_rows.append(compute_frame_features(vid_id))
        brand_rows.append(compute_brand_features(row, stt_text))
        llm_rows.append(get_llm_scalars(vid_id, llm_cache))
        cnn_rows.append(compute_cnn_features(vid_id))

        if (i + 1) % 25 == 0:
            log.info(f"  {i + 1}/{len(df)} videos processed ...")

    # Impute missing CNN rows with column mean
    valid_vecs = [v for v in cnn_rows if v is not None]
    mean_vec   = np.stack(valid_vecs).mean(axis=0)
    cnn_raw    = np.stack([v if v is not None else mean_vec for v in cnn_rows])

    return pd.DataFrame(frame_rows), pd.DataFrame(brand_rows), pd.DataFrame(llm_rows), cnn_raw


def _to_array(df_: pd.DataFrame) -> np.ndarray:
    """DataFrame → float array with NaN imputed by column mean."""
    arr      = df_.values.astype(float)
    col_mean = np.nanmean(arr, axis=0)
    mask     = np.isnan(arr)
    arr[mask] = np.take(col_mean, np.where(mask)[1])
    return arr


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(X, y_mem, y_brand, weights, groups, model_name="RF") -> dict:
    """5-fold GroupKFold (by channelName). Scaler fit inside each fold to prevent leakage."""
    results = {"mem": {"srcc": [], "pcc": [], "mse": []},
               "brand": {"srcc": [], "pcc": [], "mse": []}}

    for train_idx, val_idx in GroupKFold(n_splits=N_FOLDS).split(X, groups=groups):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[train_idx])
        X_val  = scaler.transform(X[val_idx])
        w_tr   = weights[train_idx]

        for y_all, key in [(y_mem, "mem"), (y_brand, "brand")]:
            y_tr, y_val = y_all[train_idx], y_all[val_idx]
            mdl = (BayesianRidge() if model_name == "BayesianRidge"
                   else RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED))
            mdl.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = mdl.predict(X_val)
            results[key]["srcc"].append(spearmanr(y_val, preds).correlation)
            results[key]["pcc"].append(pearsonr(y_val, preds)[0])
            results[key]["mse"].append(float(np.mean((y_val - preds) ** 2)))

    return results


def print_results(results: dict, label: str):
    bar = "═" * 64
    print(f"\n{bar}\n  {label}\n{bar}")
    for key, name in [("mem", "Video Memorability"), ("brand", "Brand Memorability")]:
        print(f"  {name}")
        print(f"    SRCC = {np.mean(results[key]['srcc']):+.4f} ± {np.std(results[key]['srcc']):.4f}")
        print(f"    PCC  = {np.mean(results[key]['pcc']):+.4f}")
        print(f"    MSE  = {np.mean(results[key]['mse']):.4f}")


def run_ablation(feature_groups: dict, y_mem, y_brand, weights, groups) -> np.ndarray:
    """Evaluate each feature group and all combined, for both BayesianRidge and RF."""
    full_X = np.concatenate(list(feature_groups.values()), axis=1)
    bar    = "═" * 64

    print(f"{bar}")
    print("  ABLATION — GroupKFold by channel (Goldman Sachs = fold 0)")
    print(bar)
    print(f"  {'Group':<28}  {'mem_SRCC':>10}  {'brand_SRCC':>10}  {'dim':>5}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*5}")

    for name, X in feature_groups.items():
        for mname in ["BayesianRidge", "RF"]:
            res   = evaluate_model(X, y_mem, y_brand, weights, groups, mname)
            label = f"{name} [{mname[:2]}]"
            print(f"  {label:<26}  {np.mean(res['mem']['srcc']):>+10.4f}  "
                  f"{np.mean(res['brand']['srcc']):>+10.4f}  {X.shape[1]:>5}")

    print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*5}")
    for mname in ["BayesianRidge", "RF"]:
        res   = evaluate_model(full_X, y_mem, y_brand, weights, groups, mname)
        label = f"ALL COMBINED [{mname[:2]}]"
        print(f"  {label:<26}  {np.mean(res['mem']['srcc']):>+10.4f}  "
              f"{np.mean(res['brand']['srcc']):>+10.4f}  {full_X.shape[1]:>5}")

    print_results(evaluate_model(full_X, y_mem, y_brand, weights, groups, "RF"),
                  "BEST: ALL COMBINED — RF (channel-stratified GroupKFold)")
    return full_X


# ── Final model training ───────────────────────────────────────────────────────

def train_final_models(X_mem, X_brand, y_mem, y_brand, weights):
    """
    Train two separate RF models on the full dataset:
      - mem model  : all 62 features
      - brand model: CNN-only 32 features (confirmed best by ablation)
    """
    def fit(X, y):
        scaler = StandardScaler()
        mdl    = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED)
        mdl.fit(scaler.fit_transform(X), y, sample_weight=weights)
        return scaler, mdl

    scaler_mem,   mdl_mem   = fit(X_mem,   y_mem)
    scaler_brand, mdl_brand = fit(X_brand, y_brand)
    log.info(f"Final models trained — mem: {X_mem.shape[1]} features, brand: {X_brand.shape[1]} features")
    return scaler_mem, scaler_brand, mdl_mem, mdl_brand


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(DATA_ROOT / "devset_videolist_GT.csv")
    log.info(f"Loaded {len(df)} videos ({df['channelName'].nunique()} channels)")

    y_mem   = df["memorability_score"].values.astype(float)
    y_brand = df["brand_memorability"].values.astype(float)
    groups  = df["channelName"].values
    weights = df["nb_annotations"].values.astype(float)
    weights /= weights.mean()  # normalise to mean=1

    llm_cache = load_llm_cache()

    log.info("Extracting features ...")
    frame_df, brand_df, llm_df, cnn_raw = extract_features(df, llm_cache)

    log.info("Applying PCA to CNN features ...")
    X_cnn, pca_model, cnn_scaler = apply_cnn_pca(cnn_raw)

    X_frame = _to_array(frame_df)
    X_brand = _to_array(brand_df)
    X_llm   = _to_array(llm_df)

    # Spearman correlation report
    bar = "═" * 64
    print(f"\n{bar}")
    print("  FEATURE → TARGET SPEARMAN CORRELATIONS  (◄ = |r| > 0.15)")
    print(bar)
    for col in pd.concat([frame_df, brand_df, llm_df], axis=1).columns:
        rm   = spearmanr(frame_df[col] if col in frame_df else
                         brand_df[col] if col in brand_df else llm_df[col], y_mem).correlation
        rb   = spearmanr(frame_df[col] if col in frame_df else
                         brand_df[col] if col in brand_df else llm_df[col], y_brand).correlation
        flag = " ◄" if (abs(rm) > 0.15 or abs(rb) > 0.15) else ""
        print(f"  {col:<32}  mem={rm:+.3f}  brand={rb:+.3f}{flag}")

    # Ablation
    log.info("Running 5-fold GroupKFold cross-validation ...")
    full_X = run_ablation(
        {"frame — dir 1": X_frame, "brand/text — dir 2": X_brand,
         "llm scalars — dir 3": X_llm, "cnn pca — dir 4": X_cnn},
        y_mem, y_brand, weights, groups
    )

    # Permutation importance (mem model, RF)
    feature_names = (list(frame_df.columns) + list(brand_df.columns) +
                     list(llm_df.columns) + [f"cnn_pca_{i}" for i in range(X_cnn.shape[1])])
    scaler_fi = StandardScaler()
    mdl_fi    = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED)
    mdl_fi.fit(scaler_fi.fit_transform(full_X), y_mem, sample_weight=weights)
    perm    = permutation_importance(mdl_fi, scaler_fi.transform(full_X), y_mem,
                                     n_repeats=20, random_state=RANDOM_SEED, scoring="r2")
    top_idx = np.argsort(perm.importances_mean)[::-1]
    print(f"\n{bar}")
    print("  TOP 15 FEATURES — permutation importance (RF, mem target)")
    print(bar)
    for rank, i in enumerate(top_idx[:15]):
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {rank+1:2d}. {fname:<38}  {perm.importances_mean[i]:+.4f}")

    # Train final models and save artefacts
    log.info("Training final models on full training set ...")
    scaler_mem, scaler_brand, mdl_mem, mdl_brand = train_final_models(
        full_X, X_cnn, y_mem, y_brand, weights)

    artefacts = {
        "mem_feature_names":   feature_names,
        "mem_scaler":          scaler_mem,
        "mdl_mem":             mdl_mem,
        "brand_feature_names": [f"cnn_pca_{i}" for i in range(X_cnn.shape[1])],
        "brand_scaler":        scaler_brand,
        "mdl_brand":           mdl_brand,
        "y_mem_train":         y_mem,       # for rank-normalisation at inference
        "y_brand_train":       y_brand,
        "pca_model":           pca_model,
        "cnn_scaler":          cnn_scaler,
        "cnn_models_used":     CNN_MODELS,
        "n_frame_features":    X_frame.shape[1],
        "n_brand_features":    X_brand.shape[1],
        "n_llm_features":      X_llm.shape[1],
        "n_cnn_features":      X_cnn.shape[1],
    }
    artefact_path = DATA_ROOT / "model_artefacts.pkl"
    with open(artefact_path, "wb") as f:
        pickle.dump(artefacts, f)
    log.info(f"Artefacts saved → {artefact_path}")
    print(f"\n{'✓ Pipeline complete.':^64}\n")


if __name__ == "__main__":
    main()