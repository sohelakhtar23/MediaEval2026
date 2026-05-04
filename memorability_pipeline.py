"""
VIDEM Memorability Prediction Pipeline
=======================================
Directions:
  1. First-minute frame features  (fixes exposure mismatch)
  2. LLM semantic scalars         (OpenAI, task-specific, low-dim)
  3. Brand entity density         (string-matching NER)
  4. Pre-extracted CNN features   (PCA-32)
  5. Annotation-weighted, channel-stratified GroupKFold evaluation

CV uses GroupKFold by channelName — prevents Goldman Sachs (23% of data)
from leaking across train/val folds, giving honest generalisation estimates.

Usage
-----
    pip install openai Pillow scikit-learn scipy pandas numpy
    export OPENAI_API_KEY=sk-...
    python memorability_pipeline.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from openai import OpenAI
from scipy.stats import spearmanr, pearsonr
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

CNN_MODELS   = ["AlexNet", "ResNet50", "VGG", "ViT",
                "EfficientNetB3", "DenseNet121", "R3D"]

OPENAI_MODEL      = "gpt-4o-mini"
MAX_STT_WORDS_LLM = 600
PCA_DIM           = 32
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
    Load first min(60, N) frames (1 frame/second) from frames_root/{video_id}/.
    Returns 14 scalar features covering colour, motion, and scene cuts.
    frames_root is parameterised so predict_test.py can point to a different
    frames folder for the test set.
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
# DIRECTION 3 — Brand entity density
# ══════════════════════════════════════════════════════════════════════════════

def compute_brand_features(row: pd.Series, stt_text: str) -> dict:
    """8 string-matching features — no external NLP library required."""
    brand = str(row.get("channelName", "")).lower().strip()
    title = str(row.get("title", "")).lower()
    desc  = str(row.get("description", "")).lower()
    tags  = str(row.get("tags", "")).lower()
    stt   = stt_text.lower()

    all_text    = " ".join([title, desc, tags, stt])
    n_words     = max(len(all_text.split()), 1)
    stt_words   = stt.split()
    n_stt_words = max(len(stt_words), 1)

    brand_count   = all_text.count(brand)
    brand_density = brand_count / n_words
    sentences     = [s.strip() for s in
                     stt.replace("?", ".").replace("!", ".").split(".")
                     if s.strip()]
    avg_sent_len  = n_stt_words / max(len(sentences), 1)
    tag_count     = len([t for t in tags.split(",") if t.strip()])

    return {
        "brand_mention_count": float(brand_count),
        "brand_density":       float(brand_density),
        "brand_in_title":      float(int(brand in title)),
        "brand_in_stt":        float(int(brand in stt)),
        "stt_word_count":      float(len(stt_words)),
        "stt_unique_ratio":    float(len(set(stt_words)) / n_stt_words),
        "avg_sentence_len":    float(avg_sent_len),
        "tag_count":           float(tag_count),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION 2 — LLM semantic scalars (OpenAI)
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
    return f"""Commercial video information:
- Brand/Channel : {row.get("channelName", "Unknown")}
- Title         : {row.get("title", "")}
- Description   : {str(row.get("description", ""))[:400]}
- Tags          : {row.get("tags", "")}
- Duration      : {row.get("durationSeconds", 0):.0f} seconds
- Transcript (~{MAX_STT_WORDS_LLM} words): {stt_clip}

Score on 8 dimensions (all floats 0-10):
{schema_lines}

Return ONLY this JSON:
{keys_json}"""


def get_llm_scalars(video_id: str, row: pd.Series, stt_text: str,
                    cache: dict, client: Optional[OpenAI]) -> dict:
    """Query OpenAI with disk-backed cache. Falls back to 5.0 if unavailable."""
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
# DIRECTION 4 — Pre-extracted CNN features (PCA-reduced)
# ══════════════════════════════════════════════════════════════════════════════

def load_cnn_features(df: pd.DataFrame,
                      features_root: Path = FEATURES_DIR,
                      pca_model: Optional[PCA] = None,
                      scaler_model: Optional[StandardScaler] = None):
    """
    Load CNN .npy features and reduce to PCA_DIM.

    Training mode  (pca_model=None): fits PCA+scaler, returns (X_pca, pca, scaler).
    Inference mode (pca_model given): transforms with existing PCA+scaler,
                                      returns (X_pca, pca_model, scaler_model).
    """
    all_mats = []
    models_used = []

    for model_name in CNN_MODELS:
        model_dir = features_root / model_name
        if not model_dir.exists():
            continue

        vecs = []
        for _, row in df.iterrows():
            vid_id = row["id"]
            npy    = model_dir / f"{vid_id}.npy"
            if not npy.exists():
                npy = model_dir / f"{int(row['video_id'])}.npy"

            if npy.exists():
                arr = np.load(npy, allow_pickle=True)
                if arr.ndim == 2:
                    v = np.concatenate([arr.mean(0), arr.std(0)])
                elif arr.ndim == 1:
                    v = arr
                else:
                    v = arr.flatten()
                vecs.append(v)
            else:
                vecs.append(None)

        n_found = sum(v is not None for v in vecs)
        if n_found < 10:
            log.info(f"  {model_name}: only {n_found} files — skipping")
            continue

        log.info(f"  {model_name}: {n_found}/{len(df)} loaded")
        valid  = [v for v in vecs if v is not None]
        mean_v = np.stack(valid).mean(axis=0)
        filled = [v if v is not None else mean_v for v in vecs]
        all_mats.append(np.stack(filled))
        models_used.append(model_name)

    if not all_mats:
        log.warning("  No CNN features found — direction 4 skipped")
        empty = np.zeros((len(df), 0))
        return empty, pca_model, scaler_model

    X_cat = np.concatenate(all_mats, axis=1)
    log.info(f"  Concatenated CNN shape: {X_cat.shape} (models: {models_used})")

    if scaler_model is None:                          # training mode
        scaler_model = StandardScaler()
        X_scaled = scaler_model.fit_transform(X_cat)
    else:                                             # inference mode
        X_scaled = scaler_model.transform(X_cat)

    n_comp = min(PCA_DIM, X_scaled.shape[1], X_scaled.shape[0] - 1)

    if pca_model is None:                             # training mode
        pca_model = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        X_pca = pca_model.fit_transform(X_scaled)
        log.info(f"  PCA-{n_comp} variance explained: "
                 f"{pca_model.explained_variance_ratio_.sum():.2%}")
    else:                                             # inference mode
        X_pca = pca_model.transform(X_scaled)

    return X_pca, pca_model, scaler_model


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — shared by train and predict
# ══════════════════════════════════════════════════════════════════════════════

def extract_handcrafted_features(df: pd.DataFrame,
                                  stt_root: Path,
                                  frames_root: Path,
                                  cache: dict,
                                  openai_client: Optional[OpenAI], is_training: bool = True) -> tuple:
    """
    Run directions 1, 2, 3 for every row in df.
    Returns (frame_df, brand_df, llm_df).
    """
    frame_rows, brand_rows, llm_rows = [], [], []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        vid_id = row["id"]

        if (i + 1) % 25 == 0:
            log.info(f"  {i + 1}/{total} videos processed ...")
            if is_training:
                with open(CACHE_FILE, "w") as fh:
                    json.dump(cache, fh, indent=2)
            else:
                cache_path    = "predict/llm_scalar_cache_test.json"
                with open(cache_path, "w") as fh:
                    json.dump(cache, fh, indent=2)

        stt_path = stt_root / f"{vid_id}.txt"
        stt_text = (stt_path.read_text(encoding="utf-8", errors="ignore")
                    if stt_path.exists() else "")

        frame_rows.append(compute_frame_features(vid_id, frames_root))
        brand_rows.append(compute_brand_features(row, stt_text))
        llm_rows.append(get_llm_scalars(vid_id, row, stt_text, cache, openai_client))

    return pd.DataFrame(frame_rows), pd.DataFrame(brand_rows), pd.DataFrame(llm_rows)


def _to_array(df_: pd.DataFrame) -> np.ndarray:
    arr      = df_.values.astype(float)
    col_mean = np.nanmean(arr, axis=0)
    mask     = np.isnan(arr)
    arr[mask] = np.take(col_mean, np.where(mask)[1])
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION 5 — Channel-stratified evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(X: np.ndarray, y_mem: np.ndarray, y_brand: np.ndarray,
                   weights: np.ndarray, groups: np.ndarray,
                   model_name: str = "RF") -> dict:
    """
    5-fold GroupKFold by channelName — channels never span train/val boundary.
    This matches the user's original stratification (Goldman Sachs = fold 0).
    """
    gkf     = GroupKFold(n_splits=N_FOLDS)
    results = {"mem":   {"srcc": [], "pcc": [], "mse": []},
               "brand": {"srcc": [], "pcc": [], "mse": []}}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        val_channels = np.unique(groups[val_idx])
        log.debug(f"  Fold {fold}: val channels = {val_channels}")

        X_tr,  X_val = X[train_idx], X[val_idx]
        w_tr         = weights[train_idx]

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
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    for key, name in [("mem", "Video Memorability"), ("brand", "Brand Memorability")]:
        srcc     = np.mean(results[key]["srcc"])
        srcc_std = np.std(results[key]["srcc"])
        pcc      = np.mean(results[key]["pcc"])
        mse      = np.mean(results[key]["mse"])
        print(f"  {name}")
        print(f"    SRCC = {srcc:+.4f} ± {srcc_std:.4f}")
        print(f"    PCC  = {pcc:+.4f}")
        print(f"    MSE  = {mse:.4f}")


def run_ablation(feature_groups: dict, y_mem, y_brand, weights, groups):
    active = {k: v for k, v in feature_groups.items() if v.shape[1] > 0}
    full_X = np.concatenate(list(active.values()), axis=1)

    bar = "═" * 64
    print(f"\n{bar}")
    print("  ABLATION — GroupKFold by channel (Goldman Sachs = fold 0)")
    print(bar)
    print(f"  {'Group':<26}  {'mem_SRCC':>10}  {'brand_SRCC':>10}  {'dim':>5}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*5}")

    for name, X in active.items():
        for mname in ["BayesianRidge", "RF"]:
            res    = evaluate_model(X, y_mem, y_brand, weights, groups, mname)
            srcc_m = np.mean(res["mem"]["srcc"])
            srcc_b = np.mean(res["brand"]["srcc"])
            label  = f"{name} [{mname[:2]}]"
            print(f"  {label:<26}  {srcc_m:>+10.4f}  {srcc_b:>+10.4f}  {X.shape[1]:>5}")

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
# FINAL MODEL — train on ALL data, save artefacts for predict_test.py
# ══════════════════════════════════════════════════════════════════════════════

def train_final_model(X: np.ndarray, y_mem: np.ndarray, y_brand: np.ndarray,
                      weights: np.ndarray) -> tuple:
    """
    Train final RF models on the full training set (no held-out fold).
    Returns (scaler, mdl_mem, mdl_brand).
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mdl_mem = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                    random_state=RANDOM_SEED)
    mdl_mem.fit(X_scaled, y_mem, sample_weight=weights)

    mdl_brand = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                      random_state=RANDOM_SEED)
    mdl_brand.fit(X_scaled, y_brand, sample_weight=weights)

    log.info("Final models trained on full training set")
    return scaler, mdl_mem, mdl_brand


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load training CSV ─────────────────────────────────────────────────────
    df = pd.read_csv(DATA_ROOT / "devset_videolist_GT.csv")
    log.info(f"Loaded {len(df)} training videos "
             f"({df['channelName'].nunique()} channels)")

    y_mem   = df["memorability_score"].values.astype(float)
    y_brand = df["brand_memorability"].values.astype(float)
    groups  = df["channelName"].values          # for GroupKFold
    weights = df["nb_annotations"].values.astype(float)
    weights = weights / weights.mean()

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

    # ── Extract handcrafted features (dirs 1, 2, 3) ───────────────────────────
    log.info("Extracting handcrafted features ...")
    frame_df, brand_df, llm_df = extract_handcrafted_features(
        df, STT_DIR, FRAMES_DIR, cache, openai_client)

    with open(CACHE_FILE, "w") as fh:
        json.dump(cache, fh, indent=2)
    log.info(f"LLM cache saved ({len(cache)} entries)")

    # ── Direction 4: CNN features ──────────────────────────────────────────────
    log.info("Loading CNN features ...")
    X_cnn, pca_model, cnn_scaler = load_cnn_features(df)

    X_frame = _to_array(frame_df)
    X_brand = _to_array(brand_df)
    X_llm   = _to_array(llm_df)

    # ── Correlation report ────────────────────────────────────────────────────
    bar = "═" * 64
    print(f"\n{bar}")
    print("  FEATURE → TARGET SPEARMAN CORRELATIONS  (◄ = |r| > 0.15)")
    print(bar)
    for col in pd.concat([frame_df, brand_df, llm_df], axis=1).columns:
        vals = pd.concat([frame_df, brand_df, llm_df], axis=1)[col]
        rm   = spearmanr(vals, y_mem).correlation
        rb   = spearmanr(vals, y_brand).correlation
        flag = " ◄" if (abs(rm) > 0.15 or abs(rb) > 0.15) else ""
        print(f"  {col:<32}  mem={rm:+.3f}  brand={rb:+.3f}{flag}")

    # ── Ablation with GroupKFold ───────────────────────────────────────────────
    feature_groups = {
        "frame — dir 1":       X_frame,
        "brand/text — dir 3":  X_brand,
        "llm scalars — dir 2": X_llm,
        "cnn pca — dir 4":     X_cnn,
    }
    log.info("Running GroupKFold cross-validation ...")
    full_X = run_ablation(feature_groups, y_mem, y_brand, weights, groups)

    # ── Permutation importance ────────────────────────────────────────────────
    print(f"\n{bar}")
    print("  TOP 15 FEATURES — permutation importance (RF, mem target)")
    print(bar)

    scaler_fi = StandardScaler()
    X_fi      = scaler_fi.fit_transform(full_X)
    mdl_fi    = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                      random_state=RANDOM_SEED)
    mdl_fi.fit(X_fi, y_mem, sample_weight=weights)
    feature_names = (list(frame_df.columns) + list(brand_df.columns) +
                     list(llm_df.columns) +
                     [f"cnn_pca_{i}" for i in range(X_cnn.shape[1])])
    perm    = permutation_importance(mdl_fi, X_fi, y_mem,
                                     n_repeats=20, random_state=RANDOM_SEED,
                                     scoring="r2")
    top_idx = np.argsort(perm.importances_mean)[::-1]
    for rank, i in enumerate(top_idx[:15]):
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {rank+1:2d}. {fname:<38}  {perm.importances_mean[i]:+.4f}")

    # ── Train final model on ALL data and save artefacts ──────────────────────
    log.info("Training final model on full training set ...")
    final_scaler, mdl_mem, mdl_brand = train_final_model(
        full_X, y_mem, y_brand, weights)

    artefacts = {
        "feature_names": feature_names,
        "final_scaler":  final_scaler,
        "mdl_mem":       mdl_mem,
        "mdl_brand":     mdl_brand,
        "pca_model":     pca_model,
        "cnn_scaler":    cnn_scaler,
        "cnn_models_used": CNN_MODELS,
    }
    artefact_path = DATA_ROOT / "model_artefacts.pkl"
    with open(artefact_path, "wb") as fh:
        pickle.dump(artefacts, fh)
    log.info(f"Artefacts saved to {artefact_path}")
    log.info("Run predict_test.py to generate test predictions.")

    print(f"\n{'✓ Pipeline complete.':^64}\n")


if __name__ == "__main__":
    main()
