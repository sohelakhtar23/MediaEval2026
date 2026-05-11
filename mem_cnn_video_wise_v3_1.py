"""
VIDEM Memorability Prediction Pipeline
=======================================
Multi-stream regression for joint prediction of video and brand memorability
on the VIDEM dataset (339 commercial videos, 32 channels).

Feature streams:
  1. First-minute frame features  — 14 visual scalars from frames[:60]
                                    (corrects annotator-exposure mismatch)
  2. Brand entity density         — 8 string-matching features from STT + metadata
  3. LLM semantic scalars         — 8 task-specific scores via OpenAI
  4. Pre-extracted CNN features   — 7 architectures, PCA-32

Model:
  - Bayesian Ridge + RF comparison; RF wins
  - annotation-count sample weighting
  - Separate models per target: mem uses all 62 features, 
    brand uses CNN-only (32 features) — confirmed by ablation
  - Channel-stratified GroupKFold (Goldman Sachs = dedicated fold)
  - Rank-normalisation at inference to correct RF mean-compression

"""

# ── Standard library ──────────────────────────────────────────────────────────
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
from PIL import Image
from openai import OpenAI
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT    = Path(".")
STT_DIR      = DATA_ROOT / "devset-stt"
FRAMES_DIR   = DATA_ROOT / "frames"
FEATURES_DIR = DATA_ROOT / "features"
CACHE_FILE   = DATA_ROOT / "llm_scalar_cache.json"

CNN_MODELS = ["AlexNet", "ResNet50", "VGG", "ViT",
              "EfficientNetB3", "DenseNet121", "R3D"]

OPENAI_MODEL      = "gpt-4o-mini"
MAX_STT_WORDS_LLM = 600
PCA_DIM           = 32    # PCA-64 was worse: noise > signal for N=339
N_FOLDS           = 5
RANDOM_SEED       = 42
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION 1 — First-minute frame features
# ══════════════════════════════════════════════════════════════════════════════

_FRAME_KEYS = [
    "frame_r_mean", "frame_g_mean", "frame_b_mean",
    "frame_r_std",  "frame_g_std",  "frame_b_std",
    "frame_brightness", "frame_bright_var",
    "frame_saturation", "frame_sat_var",
    "frame_motion",     "frame_motion_var",
    "frame_n_cuts",     "frame_color_change",
]


def _zero_frame_features() -> dict:
    return {k: 0.0 for k in _FRAME_KEYS}


def compute_frame_features(video_id: str,
                            frames_root: Path = FRAMES_DIR) -> dict:
    """
    Load the first min(60, N) frames from frames_root/{video_id}/.
    Frames are stored at 1 frame/second in ascending filename order,
    so slicing [:60] gives exactly the first minute the annotator watched.

    85.8% of VIDEM videos are longer than 60 seconds (median duration 5 min),
    so using all frames would incorporate content the annotator never saw.
    This slice directly corrects that exposure mismatch.

    Returns 14 scalars: RGB channel statistics, brightness, saturation,
    inter-frame motion energy, scene cut count, and temporal colour change.
    """
    frame_dir = frames_root / video_id
    if not frame_dir.exists():
        log.warning(f"  No frames dir: {video_id}")
        return _zero_frame_features()

    frames_to_use = sorted(frame_dir.iterdir())[:60]
    if not frames_to_use:
        return _zero_frame_features()

    pixel_means, pixel_stds = [], []
    brightness_vals, sat_vals, diffs = [], [], []
    prev_arr = None

    for fp in frames_to_use:
        try:
            arr = np.array(
                Image.open(fp).convert("RGB").resize((64, 64)),
                dtype=np.float32
            ) / 255.0
            pixel_means.append(arr.mean(axis=(0, 1)))
            pixel_stds.append(arr.std(axis=(0, 1)))
            brightness_vals.append(arr.mean())
            sat_vals.append(arr.std(axis=2).mean())
            if prev_arr is not None:
                diffs.append(np.abs(arr - prev_arr).mean())
            prev_arr = arr
        except Exception as exc:
            log.debug(f"    Skipping {fp.name}: {exc}")

    if not pixel_means:
        return _zero_frame_features()

    pixel_means = np.stack(pixel_means)
    pixel_stds  = np.stack(pixel_stds)

    return {
        "frame_r_mean":       float(pixel_means[:, 0].mean()),
        "frame_g_mean":       float(pixel_means[:, 1].mean()),
        "frame_b_mean":       float(pixel_means[:, 2].mean()),
        "frame_r_std":        float(pixel_stds[:, 0].mean()),
        "frame_g_std":        float(pixel_stds[:, 1].mean()),
        "frame_b_std":        float(pixel_stds[:, 2].mean()),
        "frame_brightness":   float(np.mean(brightness_vals)),
        "frame_bright_var":   float(np.std(brightness_vals)),
        "frame_saturation":   float(np.mean(sat_vals)),
        "frame_sat_var":      float(np.std(sat_vals)),
        "frame_motion":       float(np.mean(diffs)) if diffs else 0.0,
        "frame_motion_var":   float(np.std(diffs))  if diffs else 0.0,
        "frame_n_cuts":       float(sum(1 for d in diffs if d > 0.05)),
        "frame_color_change": float(pixel_means.std(axis=0).mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION 2 — Brand entity density
# ══════════════════════════════════════════════════════════════════════════════

def compute_brand_features(row: pd.Series, stt_text: str) -> dict:
    """
    8 string-matching brand and text features requiring no external NLP library.
    Covers brand exposure (mention count, density, title/STT presence),
    STT richness (word count, type-token ratio, sentence length), and tag count.
    """
    brand = str(row.get("channelName", "")).lower().strip()
    title = str(row.get("title", "")).lower()
    desc  = str(row.get("description", "")).lower()
    tags  = str(row.get("tags", "")).lower()
    stt   = stt_text.lower()

    all_text    = " ".join([title, desc, tags, stt])
    n_words     = max(len(all_text.split()), 1)
    stt_words   = stt.split()
    n_stt_words = max(len(stt_words), 1)

    brand_count  = all_text.count(brand)
    sentences    = [s.strip() for s in
                    stt.replace("?", ".").replace("!", ".").split(".")
                    if s.strip()]
    avg_sent_len = n_stt_words / max(len(sentences), 1)
    tag_count    = len([t for t in tags.split(",") if t.strip()])

    return {
        "brand_mention_count": float(brand_count),
        "brand_density":       float(brand_count / n_words),
        "brand_in_title":      float(int(brand in title)),
        "brand_in_stt":        float(int(brand in stt)),
        "stt_word_count":      float(len(stt_words)),
        "stt_unique_ratio":    float(len(set(stt_words)) / n_stt_words),
        "avg_sentence_len":    float(avg_sent_len),
        "tag_count":           float(tag_count),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION 3 — LLM semantic scalars (OpenAI)
# ══════════════════════════════════════════════════════════════════════════════

_LLM_SYSTEM = (
    "You are an expert in advertising psychology and video memorability. "
    "Given a commercial video's metadata and transcript excerpt, score it on "
    "8 dimensions. Return ONLY a valid JSON object — no markdown, no commentary."
)

_LLM_DIMENSIONS = {
    "brand_prominence":    "How centrally/repeatedly is the brand name featured? (0=never, 10=constant)",
    "emotional_valence":   "Emotional tone: 0=cold/alarming/negative → 10=warm/uplifting/positive",
    "narrative_arc":       "Clear story structure (problem→solution)? 0=none, 10=strong arc",
    "call_to_action":      "How explicit is the call to action? 0=none, 10=very direct",
    "information_density": "Factual/technical load: 0=light/emotional, 10=very dense/technical",
    "novelty_surprise":    "How unexpected/counter-intuitive is the message? 0=predictable, 10=surprising",
    "visual_dynamism":     "Inferred visual energy: 0=static/talking-head, 10=high-action/variety",
    "brand_specificity":   "Concrete products/numbers vs vague positioning: 0=vague, 10=very specific",
}

_LLM_KEYS = list(_LLM_DIMENSIONS.keys())


def _default_llm_scalars() -> dict:
    return {k: 5.0 for k in _LLM_KEYS}


def _build_llm_prompt(row: pd.Series, stt_text: str) -> str:
    stt_clip     = " ".join(stt_text.split()[:MAX_STT_WORDS_LLM])
    schema_lines = "\n".join(f'  "{k}": {v}' for k, v in _LLM_DIMENSIONS.items())
    keys_json    = "{" + ", ".join(f'"{k}": ...' for k in _LLM_KEYS) + "}"
    return (
        f"Commercial video information:\n"
        f"- Brand/Channel : {row.get('channelName', 'Unknown')}\n"
        f"- Title         : {row.get('title', '')}\n"
        f"- Description   : {str(row.get('description', ''))[:400]}\n"
        f"- Tags          : {row.get('tags', '')}\n"
        f"- Duration      : {row.get('durationSeconds', 0):.0f} seconds\n"
        f"- Transcript (~{MAX_STT_WORDS_LLM} words): {stt_clip}\n\n"
        f"Score this commercial on 8 dimensions (all floats 0-10):\n"
        f"{schema_lines}\n\n"
        f"Return ONLY this JSON (no extra text):\n"
        f"{keys_json}"
    )


def get_llm_scalars(video_id: str, row: pd.Series, stt_text: str,
                    cache: dict, client: Optional[OpenAI]) -> dict:
    """
    Query OpenAI for 8 task-specific semantic scalars with disk-backed caching.
    response_format=json_object guarantees valid JSON output.
    Falls back to neutral 5.0 values if no API key is available.
    """
    if video_id in cache:
        return cache[video_id]
    if client is None:
        return _default_llm_scalars()
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=256,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": _build_llm_prompt(row, stt_text)},
            ],
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        result = {k: float(parsed.get(k, 5.0)) for k in _LLM_KEYS}
        cache[video_id] = result
        return result
    except Exception as exc:
        log.warning(f"  LLM call failed for {video_id}: {exc} — using neutral")
        return _default_llm_scalars()


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION 4 — Pre-extracted CNN features (per-video, PCA-32)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cnn_features_for_video(vid_id: str,
                                    features_root: Path = FEATURES_DIR) -> Optional[np.ndarray]:
    """
    Load pre-extracted CNN/Transformer features for a single video and
    return one concatenated raw feature vector.

    Per-model collapse:
      - 2-D array (T, D)  → mean+std concatenation  (length 2*D)
      - 1-D array (D,)    → used as-is
      - N-D array         → flattened

    Returns None only if no model file is found for this video.
    """
    parts = []
    for model_name in CNN_MODELS:
        npy = features_root / model_name / f"{vid_id}.npy"
        if not npy.exists():
            log.debug(f"  CNN [{model_name}] missing for {vid_id} — skipped")
            continue
        arr = np.load(npy, allow_pickle=True)
        if arr.ndim == 2:
            v = np.concatenate([arr.mean(0), arr.std(0)])
        elif arr.ndim == 1:
            v = arr
        else:
            v = arr.flatten()
        parts.append(v)

    if not parts:
        return None
    return np.concatenate(parts)


def apply_cnn_pca(cnn_raw: np.ndarray,
                  pca_model: Optional[PCA] = None,
                  scaler_model: Optional[StandardScaler] = None):
    """
    Scale and PCA-reduce the (N, raw_dim) CNN matrix built by the per-video loop.

    Training mode  (models=None): fits scaler+PCA, returns (X_pca, pca, scaler).
    Inference mode (models given): transforms only,  returns same tuple.
    """
    log.info(f"  Concatenated CNN shape: {cnn_raw.shape}")

    if scaler_model is None:                          # training mode
        scaler_model = StandardScaler()
        X_scaled = scaler_model.fit_transform(cnn_raw)
    else:                                             # inference mode
        X_scaled = scaler_model.transform(cnn_raw)

    if pca_model is None:                             # training mode
        pca_model = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
        X_pca = pca_model.fit_transform(X_scaled)
        log.info(f"  PCA-{PCA_DIM} variance explained: "
                 f"{pca_model.explained_variance_ratio_.sum():.2%}")
    else:                                             # inference mode
        X_pca = pca_model.transform(X_scaled)

    return X_pca, pca_model, scaler_model


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — shared by train and predict
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(df: pd.DataFrame,
                     stt_root: Path,
                     frames_root: Path,
                     cache: dict,
                     openai_client: Optional[OpenAI],
                     features_root: Path = FEATURES_DIR,
                     is_training: bool = True) -> tuple:
    """
    Run all four feature directions for every video in df.
    Returns (frame_df, brand_df, llm_df, cnn_raw_array).

    cnn_raw_array is the (N, raw_dim) matrix before PCA reduction;
    pass it to apply_cnn_pca() to obtain the final PCA-32 features.
    """
    frame_rows, brand_rows, llm_rows, cnn_rows = [], [], [], []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        vid_id = row["id"]
        if (i + 1) % 25 == 0:
            log.info(f"  {i + 1}/{total} videos processed ...")
            cache_path = CACHE_FILE if is_training else Path("predict/llm_scalar_cache_test.json")
            with open(cache_path, "w") as fh:
                json.dump(cache, fh, indent=2)

        stt_path = stt_root / f"{vid_id}.txt"
        stt_text = (stt_path.read_text(encoding="utf-8", errors="ignore")
                    if stt_path.exists() else "")

        frame_rows.append(compute_frame_features(vid_id, frames_root))
        brand_rows.append(compute_brand_features(row, stt_text))
        llm_rows.append(get_llm_scalars(vid_id, row, stt_text, cache, openai_client))
        cnn_rows.append(compute_cnn_features_for_video(vid_id, features_root))

    # Assemble CNN raw matrix; impute any missing rows with the column mean
    valid_vecs = [v for v in cnn_rows if v is not None]
    raw_dim    = valid_vecs[0].shape[0] if valid_vecs else 0
    mean_vec   = np.stack(valid_vecs).mean(axis=0) if valid_vecs else np.zeros(raw_dim)
    cnn_raw    = np.stack([v if v is not None else mean_vec for v in cnn_rows])

    return (pd.DataFrame(frame_rows),
            pd.DataFrame(brand_rows),
            pd.DataFrame(llm_rows),
            cnn_raw)


def _to_array(df_: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to float array, imputing NaNs with column means."""
    arr      = df_.values.astype(float)
    col_mean = np.nanmean(arr, axis=0)
    mask     = np.isnan(arr)
    arr[mask] = np.take(col_mean, np.where(mask)[1])
    return arr



# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — Channel-stratified GroupKFold
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(X: np.ndarray, y_mem: np.ndarray, y_brand: np.ndarray,
                   weights: np.ndarray, groups: np.ndarray,
                   model_name: str = "RF") -> dict:
    """
    5-fold GroupKFold by channelName.
    Goldman Sachs (79 videos, 23% of data) lands exclusively in fold 0,
    preventing its vocabulary and style from leaking into validation.
    Scaler is fit inside each fold to prevent data leakage.
    """
    gkf     = GroupKFold(n_splits=N_FOLDS)
    results = {"mem":   {"srcc": [], "pcc": [], "mse": []},
               "brand": {"srcc": [], "pcc": [], "mse": []}}

    for train_idx, val_idx in gkf.split(X, groups=groups):
        X_tr, X_val = X[train_idx], X[val_idx]
        w_tr        = weights[train_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        for y_all, key in [(y_mem, "mem"), (y_brand, "brand")]:
            y_tr  = y_all[train_idx]
            y_val = y_all[val_idx]

            if model_name == "BayesianRidge":
                mdl = BayesianRidge()
            else:
                mdl = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                            random_state=RANDOM_SEED)
            mdl.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = mdl.predict(X_val)

            results[key]["srcc"].append(spearmanr(y_val, preds).correlation)
            results[key]["pcc"].append(pearsonr(y_val, preds)[0])
            results[key]["mse"].append(float(np.mean((y_val - preds) ** 2)))

    return results


def print_results(results: dict, label: str):
    bar = "═" * 64
    print(f"\n{bar}\n  {label}\n{bar}")
    for key, name in [("mem", "Video Memorability"),
                      ("brand", "Brand Memorability")]:
        srcc     = np.mean(results[key]["srcc"])
        srcc_std = np.std(results[key]["srcc"])
        pcc      = np.mean(results[key]["pcc"])
        mse      = np.mean(results[key]["mse"])
        print(f"  {name}")
        print(f"    SRCC = {srcc:+.4f} ± {srcc_std:.4f}")
        print(f"    PCC  = {pcc:+.4f}")
        print(f"    MSE  = {mse:.4f}")


def run_ablation(feature_groups: dict, y_mem, y_brand, weights, groups):
    """Evaluate each feature group alone, then all combined."""
    active = {k: v for k, v in feature_groups.items() if v.shape[1] > 0}
    full_X = np.concatenate(list(active.values()), axis=1)

    bar = "═" * 64
    print(f"{bar}")
    print("  ABLATION — GroupKFold by channel (Goldman Sachs = fold 0)")
    print(bar)
    print(f"  {'Group':<28}  {'mem_SRCC':>10}  {'brand_SRCC':>10}  {'dim':>5}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*5}")

    # ablation: each group alone
    for name, X in active.items():
        for mname in ["BayesianRidge", "RF"]:
            res    = evaluate_model(X, y_mem, y_brand, weights, groups, mname)
            srcc_m = np.mean(res["mem"]["srcc"])
            srcc_b = np.mean(res["brand"]["srcc"])
            label  = f"{name} [{mname[:2]}]"
            print(f"  {label:<26}  {srcc_m:>+10.4f}  {srcc_b:>+10.4f}  {X.shape[1]:>5}")
    # all features combined
    print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*5}")
    for mname in ["BayesianRidge", "RF"]:
        res_full = evaluate_model(full_X, y_mem, y_brand, weights, groups, mname)
        srcc_m   = np.mean(res_full["mem"]["srcc"])
        srcc_b   = np.mean(res_full["brand"]["srcc"])
        label    = f"ALL COMBINED [{mname[:2]}]"
        print(f"  {label:<26}  {srcc_m:>+10.4f}  {srcc_b:>+10.4f}  {full_X.shape[1]:>5}")

    res_best = evaluate_model(full_X, y_mem, y_brand, weights, groups, "RF")
    print_results(res_best, "BEST: ALL COMBINED — RF (channel-stratified GroupKFold)")
    return full_X


# ══════════════════════════════════════════════════════════════════════════════
# FINAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_final_models(X_mem: np.ndarray, X_brand: np.ndarray,
                       y_mem: np.ndarray, y_brand: np.ndarray,
                       weights: np.ndarray) -> tuple:
    """
    Train separate RF models (300 trees) per target on the full dataset.
    Mem   model: all 62 features.
    Brand model: CNN-only (32 features) — confirmed best by ablation.
    Returns (scaler_mem, scaler_brand, mdl_mem, mdl_brand).
    """
    scaler_mem   = StandardScaler()
    mdl_mem      = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                         random_state=RANDOM_SEED)
    mdl_mem.fit(scaler_mem.fit_transform(X_mem), y_mem, sample_weight=weights)

    scaler_brand = StandardScaler()
    mdl_brand    = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                         random_state=RANDOM_SEED)
    mdl_brand.fit(scaler_brand.fit_transform(X_brand), y_brand,
                  sample_weight=weights)

    log.info(f"Final models trained — mem: {X_mem.shape[1]} features, "
             f"brand: {X_brand.shape[1]} features")
    return scaler_mem, scaler_brand, mdl_mem, mdl_brand


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_ROOT / "devset_videolist_GT.csv")
    log.info(f"Loaded {len(df)} training videos "
             f"({df['channelName'].nunique()} channels)")

    y_mem   = df["memorability_score"].values.astype(float)
    y_brand = df["brand_memorability"].values.astype(float)
    groups  = df["channelName"].values
    weights = df["nb_annotations"].values.astype(float)
    weights = weights / weights.mean()      # normalise to mean=1

    # ── OpenAI client ─────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None
    if not openai_client:
        log.warning("OPENAI_API_KEY not set — LLM scalars will be neutral (5.0)")

    cache: dict = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as fh:
            cache = json.load(fh)
        log.info(f"LLM cache: {len(cache)} entries loaded")

    # ── Feature extraction — all 4 directions, per-video ─────────────────────
    log.info("Extracting features (all directions, per-video) ...")
    frame_df, brand_df, llm_df, cnn_raw = extract_features(
        df, STT_DIR, FRAMES_DIR, cache, openai_client, FEATURES_DIR)

    with open(CACHE_FILE, "w") as fh:
        json.dump(cache, fh, indent=2)
    log.info(f"LLM cache saved ({len(cache)} entries)")

    log.info("Applying PCA to CNN features (dir 4) ...")
    X_cnn, pca_model, cnn_scaler = apply_cnn_pca(cnn_raw)

    X_frame = _to_array(frame_df)
    X_brand = _to_array(brand_df)
    X_llm   = _to_array(llm_df)

    # ── Spearman correlation report ───────────────────────────────────────────
    bar = "═" * 64
    print(f"\n{bar}")
    print("  FEATURE → TARGET SPEARMAN CORRELATIONS  (◄ = |r| > 0.15)")
    print(bar)
    handcrafted = pd.concat([frame_df, brand_df, llm_df], axis=1)
    for col in handcrafted.columns:
        rm   = spearmanr(handcrafted[col], y_mem).correlation
        rb   = spearmanr(handcrafted[col], y_brand).correlation
        flag = " ◄" if (abs(rm) > 0.15 or abs(rb) > 0.15) else ""
        print(f"  {col:<32}  mem={rm:+.3f}  brand={rb:+.3f}{flag}")
    print()

    # ── Ablation (all features) ───────────────────────────────────────────────
    feature_groups = {
        "frame — dir 1":       X_frame,
        "brand/text — dir 2":  X_brand,
        "llm scalars — dir 3": X_llm,
        "cnn pca — dir 4":     X_cnn,
    }
    log.info("Running 5-fold GroupKFold cross-validation ...")
    full_X = run_ablation(feature_groups, y_mem, y_brand, weights, groups)

    # ── Permutation importance (mem model) ───────────────────────────────────
    print(f"\n{bar}")
    print("  TOP 15 FEATURES — permutation importance (RF, mem target)")
    print(bar)

    feature_names = (list(frame_df.columns) + list(brand_df.columns) +
                     list(llm_df.columns) +
                     [f"cnn_pca_{i}" for i in range(X_cnn.shape[1])])

    scaler_fi = StandardScaler()
    X_fi      = scaler_fi.fit_transform(full_X)
    mdl_fi    = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                      random_state=RANDOM_SEED)
    mdl_fi.fit(X_fi, y_mem, sample_weight=weights)
    perm    = permutation_importance(mdl_fi, X_fi, y_mem,
                                     n_repeats=20, random_state=RANDOM_SEED,
                                     scoring="r2")
    top_idx = np.argsort(perm.importances_mean)[::-1]
    for rank, i in enumerate(top_idx[:15]):
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {rank+1:2d}. {fname:<38}  {perm.importances_mean[i]:+.4f}")


    # ── Train final models on full dataset ────────────────────────────────────
    log.info("Training final models on full training set ...")
    scaler_mem, scaler_brand, mdl_mem, mdl_brand = train_final_models(
        full_X, X_cnn, y_mem, y_brand, weights)

    # ── Save artefacts ────────────────────────────────────────────────────────
    artefacts = {
        # Mem model — all 62 features
        "mem_feature_names": feature_names,
        "mem_scaler":        scaler_mem,
        "mdl_mem":           mdl_mem,

        # Brand model — CNN-only (32 features)
        "brand_feature_names": [f"cnn_pca_{i}" for i in range(X_cnn.shape[1])],
        "brand_scaler":        scaler_brand,
        "mdl_brand":           mdl_brand,

        # Training distributions for rank-normalisation at inference
        "y_mem_train":   y_mem,
        "y_brand_train": y_brand,

        # CNN transform (shared between mem and brand)
        "pca_model":       pca_model,
        "cnn_scaler":      cnn_scaler,
        "cnn_models_used": CNN_MODELS,

        # Metadata
        "n_frame_features": X_frame.shape[1],
        "n_brand_features": X_brand.shape[1],
        "n_llm_features":   X_llm.shape[1],
        "n_cnn_features":   X_cnn.shape[1],
    }
    artefact_path = DATA_ROOT / "model_artefacts.pkl"
    with open(artefact_path, "wb") as fh:
        pickle.dump(artefacts, fh)

    log.info(f"Artefacts saved → {artefact_path}")
    log.info(f"  Mem model   : all features  (dim={full_X.shape[1]})")
    log.info(f"  Brand model : CNN-only       (dim={X_cnn.shape[1]})")
    log.info("Run predict_test.py to generate test predictions.")
    print(f"\n{'✓ Pipeline complete.':^64}\n")


if __name__ == "__main__":
    main()