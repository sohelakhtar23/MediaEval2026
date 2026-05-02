"""
VIDEM Memorability Prediction Pipeline
=======================================
Five research directions combined into a single pipeline:
  1. First-minute frame features  (fixes exposure mismatch — 85% of videos >60s)
  2. LLM semantic scalars         (task-specific, low-dim, via OpenAI)
  3. Brand entity density         (string-matching NER on STT + metadata)
  4. Pre-extracted CNN features   (PCA-32 reduction, safe for N=339)
  5. Annotation-weighted multi-output Bayesian Ridge + RF comparison

Usage
-----
    pip install openai Pillow scikit-learn scipy pandas numpy
    export OPENAI_API_KEY=sk-...
    python memorability_pipeline.py

Directory layout expected (all relative to DATA_ROOT):
    devset_videolist_GT.csv
    devset-stt/{id}.txt
    frames/{id}/*.jpg        (sorted ascending = chronological, 1 frame/second)
    features/AlexNet/{id}.npy   (and other CNN feature sub-folders)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from PIL import Image
from openai import OpenAI
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these paths if your layout differs
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT    = Path(".")                          # folder that contains the CSV
STT_DIR      = DATA_ROOT / "devset-stt"           # {id}.txt files
FRAMES_DIR   = DATA_ROOT / "frames"               # frames/{id}/*.jpg
FEATURES_DIR = DATA_ROOT / "features"             # features/AlexNet/{id}.npy
CACHE_FILE   = DATA_ROOT / "llm_scalar_cache.json"  # persists API responses

CNN_MODELS   = ["AlexNet", "ResNet50", "VGG", "ViT",
                "EfficientNetB3", "DenseNet121", "R3D"]

OPENAI_MODEL      = "gpt-4o-mini"   # cheap + reliable JSON mode
MAX_STT_WORDS_LLM = 600             # words fed to LLM (keep prompt short)
PCA_DIM           = 32              # CNN reduction dim (safe for N=339)
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


def compute_frame_features(video_id: str) -> dict:
    """
    Load only the first min(60, N) frames from frames/{video_id}/.
    1 frame per second, sorted ascending = chronological, so this slice
    is exactly the first minute the annotator watched.

    Returns 14 scalar features: colour (RGB mean/std), brightness,
    saturation, motion energy, scene cuts, temporal colour change.
    """
    frame_dir = FRAMES_DIR / video_id
    if not frame_dir.exists():
        log.warning(f"  No frames dir: {video_id}")
        return _zero_frame_features()

    all_frames    = sorted(frame_dir.iterdir())   # ascending = chronological
    frames_to_use = all_frames[:60]               # first-minute slice
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
            ) / 255.0                                       # (64, 64, 3)

            pixel_means.append(arr.mean(axis=(0, 1)))      # (3,)
            pixel_stds.append(arr.std(axis=(0, 1)))        # (3,)
            brightness_vals.append(arr.mean())
            sat_vals.append(arr.std(axis=2).mean())        # inter-channel std proxy

            if prev_arr is not None:
                diffs.append(np.abs(arr - prev_arr).mean())
            prev_arr = arr
        except Exception as exc:
            log.debug(f"    Skipping {fp.name}: {exc}")

    if not pixel_means:
        return _zero_frame_features()

    pixel_means = np.stack(pixel_means)            # (T, 3)
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
    """
    String-matching brand features — no external NLP library required.
    Returns 8 scalars covering brand exposure, STT richness, and metadata.
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

    brand_count   = all_text.count(brand)
    brand_density = brand_count / n_words

    # Sentence count as information-density proxy
    sentences    = [s.strip() for s in
                    stt.replace("?", ".").replace("!", ".").split(".")
                    if s.strip()]
    avg_sent_len = n_stt_words / max(len(sentences), 1)

    tag_count = len([t for t in tags.split(",") if t.strip()])

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
    """Neutral fallback (5.0) when API is unavailable."""
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

Score this commercial on 8 dimensions (all floats 0-10):
{schema_lines}

Return ONLY this JSON (no extra text):
{keys_json}"""


def get_llm_scalars(video_id: str, row: pd.Series, stt_text: str,
                    cache: dict, client: Optional[OpenAI]) -> dict:
    """
    Query OpenAI for 8 semantic scalars with a disk-backed cache.
    response_format=json_object guarantees valid JSON back from the API.
    Falls back to neutral 5.0 values if no client is available.
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
            response_format={"type": "json_object"},   # enforces valid JSON output
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

def load_cnn_features(df: pd.DataFrame) -> np.ndarray:
    """
    Load available pre-extracted CNN/Transformer .npy features.
    Collapses any temporal (T, D) dimension with mean+std concatenation,
    then reduces all models jointly to PCA_DIM components.
    Returns (N, PCA_DIM) or (N, 0) if no feature files are found.
    """
    all_mats = []

    for model_name in CNN_MODELS:
        model_dir = FEATURES_DIR / model_name
        if not model_dir.exists():
            continue

        vecs = []
        for _, row in df.iterrows():
            vid_id = row["id"]
            npy    = model_dir / f"{vid_id}.npy"
            if not npy.exists():                           # fallback: integer filename
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
            log.info(f"  {model_name}: only {n_found} files found — skipping")
            continue

        log.info(f"  {model_name}: {n_found}/{len(df)} videos loaded")
        valid  = [v for v in vecs if v is not None]
        mean_v = np.stack(valid).mean(axis=0)
        filled = [v if v is not None else mean_v for v in vecs]
        all_mats.append(np.stack(filled))               # (N, D_model)

    if not all_mats:
        log.warning("  No CNN features found — direction 4 skipped")
        return np.zeros((len(df), 0))

    X_cat  = np.concatenate(all_mats, axis=1)           # (N, total_D)
    log.info(f"  Concatenated CNN shape: {X_cat.shape}")

    X_scaled = StandardScaler().fit_transform(X_cat)
    n_comp   = min(PCA_DIM, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca      = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    X_pca    = pca.fit_transform(X_scaled)
    log.info(f"  PCA-{n_comp} variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    return X_pca


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION 5 — Annotation-weighted multi-output evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(X: np.ndarray, y_mem: np.ndarray, y_brand: np.ndarray,
                   weights: np.ndarray, model_name: str = "BayesianRidge") -> dict:
    """
    5-fold CV reporting SRCC / PCC / MSE for both targets.
    sample_weight=nb_annotations applied only during training (not evaluation).
    Scaler is fit inside each fold to prevent data leakage.
    """
    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {"mem":   {"srcc": [], "pcc": [], "mse": []},
               "brand": {"srcc": [], "pcc": [], "mse": []}}

    for train_idx, val_idx in kf.split(X):
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
                mdl = RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                            random_state=RANDOM_SEED)
            mdl.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = mdl.predict(X_val)

            results[key]["srcc"].append(spearmanr(y_val, preds).correlation)
            results[key]["pcc"].append(pearsonr(y_val, preds)[0])
            results[key]["mse"].append(float(np.mean((y_val - preds) ** 2)))

    return results


def print_results(results: dict, label: str):
    bar = "═" * 62
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


def run_ablation(feature_groups: dict, y_mem: np.ndarray, y_brand: np.ndarray,
                 weights: np.ndarray):
    """
    Evaluate each feature group individually, then all combined.
    Prints an aligned ablation table. Returns (full_X, full_results).
    """
    active = {k: v for k, v in feature_groups.items() if v.shape[1] > 0}
    full_X = np.concatenate(list(active.values()), axis=1)

    bar = "═" * 62
    print(f"\n{bar}")
    print("  ABLATION — individual feature groups (BayesianRidge, 5-fold CV)")
    print(bar)
    print(f"  {'Group':<26}  {'mem_SRCC':>10}  {'brand_SRCC':>10}  {'dim':>5}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*5}")

    for name, X in active.items():
        res    = evaluate_model(X, y_mem, y_brand, weights)
        srcc_m = np.mean(res["mem"]["srcc"])
        srcc_b = np.mean(res["brand"]["srcc"])
        print(f"  {name:<26}  {srcc_m:>+10.4f}  {srcc_b:>+10.4f}  {X.shape[1]:>5}")

    res_full = evaluate_model(full_X, y_mem, y_brand, weights)
    srcc_m   = np.mean(res_full["mem"]["srcc"])
    srcc_b   = np.mean(res_full["brand"]["srcc"])
    print(f"  {'ALL COMBINED':<26}  {srcc_m:>+10.4f}  {srcc_b:>+10.4f}  {full_X.shape[1]:>5}")

    print_results(res_full, "FULL MODEL — BayesianRidge (annotation-weighted)")

    res_rf = evaluate_model(full_X, y_mem, y_brand, weights, model_name="RF")
    print_results(res_rf, "FULL MODEL — Random Forest (annotation-weighted)")

    return full_X, res_full


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load CSV ───────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_ROOT / "devset_videolist_GT.csv")
    log.info(f"Loaded {len(df)} videos")

    y_mem   = df["memorability_score"].values.astype(float)
    y_brand = df["brand_memorability"].values.astype(float)

    # Normalise annotation counts to mean=1 (stabilises BayesianRidge numerics)
    weights = df["nb_annotations"].values.astype(float)
    weights = weights / weights.mean()

    # ── OpenAI client (optional — pipeline runs without it) ───────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        openai_client: Optional[OpenAI] = OpenAI(api_key=api_key)
        log.info(f"OpenAI client ready (model: {OPENAI_MODEL})")
    else:
        openai_client = None
        log.warning("OPENAI_API_KEY not set — LLM scalars will be neutral (5.0)")

    # ── Load LLM cache ─────────────────────────────────────────────────────────
    cache: dict = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as fh:
            cache = json.load(fh)
        log.info(f"LLM cache loaded: {len(cache)} entries")

    # ── Feature extraction loop ────────────────────────────────────────────────
    frame_rows, brand_rows, llm_rows = [], [], []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        vid_id = row["id"]

        if (i + 1) % 25 == 0:
            log.info(f"  {i + 1}/{total} videos processed ...")
            with open(CACHE_FILE, "w") as fh:       # periodic cache flush
                json.dump(cache, fh, indent=2)

        stt_path = STT_DIR / f"{vid_id}.txt"
        stt_text = (stt_path.read_text(encoding="utf-8", errors="ignore")
                    if stt_path.exists() else "")

        frame_rows.append(compute_frame_features(vid_id))           # direction 1
        brand_rows.append(compute_brand_features(row, stt_text))    # direction 3
        llm_rows.append(                                             # direction 2
            get_llm_scalars(vid_id, row, stt_text, cache, openai_client))

    with open(CACHE_FILE, "w") as fh:
        json.dump(cache, fh, indent=2)
    log.info(f"LLM cache saved ({len(cache)} entries)")

    frame_df = pd.DataFrame(frame_rows)
    brand_df = pd.DataFrame(brand_rows)
    llm_df   = pd.DataFrame(llm_rows)

    log.info(f"Frame features : {frame_df.shape}")
    log.info(f"Brand features : {brand_df.shape}")
    log.info(f"LLM scalars    : {llm_df.shape}")

    # ── Direction 4: CNN features ──────────────────────────────────────────────
    log.info("Loading pre-extracted CNN features ...")
    X_cnn = load_cnn_features(df)

    # ── Spearman correlation report ────────────────────────────────────────────
    bar = "═" * 62
    print(f"\n{bar}")
    print("  FEATURE → TARGET SPEARMAN CORRELATIONS  (◄ = |r| > 0.15)")
    print(bar)
    handcrafted_df = pd.concat([frame_df, brand_df, llm_df], axis=1)
    for col in handcrafted_df.columns:
        rm   = spearmanr(handcrafted_df[col], y_mem).correlation
        rb   = spearmanr(handcrafted_df[col], y_brand).correlation
        flag = " ◄" if (abs(rm) > 0.15 or abs(rb) > 0.15) else ""
        print(f"  {col:<32}  mem={rm:+.3f}  brand={rb:+.3f}{flag}")

    # ── Assemble feature matrices ──────────────────────────────────────────────
    def to_array(df_: pd.DataFrame) -> np.ndarray:
        arr      = df_.values.astype(float)
        col_mean = np.nanmean(arr, axis=0)
        mask     = np.isnan(arr)
        arr[mask] = np.take(col_mean, np.where(mask)[1])
        return arr

    X_frame = to_array(frame_df)
    X_brand = to_array(brand_df)
    X_llm   = to_array(llm_df)

    feature_groups = {
        "frame — dir 1":       X_frame,
        "brand/text — dir 3":  X_brand,
        "llm scalars — dir 2": X_llm,
        "cnn pca — dir 4":     X_cnn,
    }

    # ── Ablation + final evaluation ────────────────────────────────────────────
    log.info("Running 5-fold cross-validation ...")
    full_X, _ = run_ablation(feature_groups, y_mem, y_brand, weights)

    # ── Permutation feature importance ─────────────────────────────────────────
    print(f"\n{bar}")
    print("  TOP 15 FEATURES — permutation importance (BayesianRidge, mem target)")
    print(bar)

    X_scaled = StandardScaler().fit_transform(full_X)
    mdl      = BayesianRidge()
    mdl.fit(X_scaled, y_mem, sample_weight=weights)

    feature_names = (list(frame_df.columns) +
                     list(brand_df.columns) +
                     list(llm_df.columns) +
                     [f"cnn_pca_{i}" for i in range(X_cnn.shape[1])])

    perm    = permutation_importance(mdl, X_scaled, y_mem,
                                     n_repeats=20, random_state=RANDOM_SEED,
                                     scoring="r2")
    top_idx = np.argsort(perm.importances_mean)[::-1]
    for rank, i in enumerate(top_idx[:15]):
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {rank + 1:2d}. {fname:<38}  {perm.importances_mean[i]:+.4f}")

    print(f"\n{'✓ Pipeline complete.':^62}\n")


if __name__ == "__main__":
    main()