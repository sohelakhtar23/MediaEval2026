"""
Direction 1: CLIP Zero-Shot Frame Scoring
- Loads first 60 frames per video
- Scores each frame against memorability & brand prompts
- Aggregates into per-video scalar features
- Saves results to clip_features.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy.stats import linregress
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

FRAMES_DIR   = "frames"          # frames/{video_id}/*.jpg
METADATA_CSV = "devset_videolist_GT.csv"
OUTPUT_CSV   = "clip_features.csv"

MAX_FRAMES   = 60                # first 60 seconds = what annotators watched
BATCH_SIZE   = 32                # fits comfortably in 20 GB VRAM
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Prompts ───────────────────────────────────────────────────────────────────

VIDEO_MEM_PROMPTS = [
    "a memorable advertisement",
    "an emotionally engaging video",
    "a surprising or striking scene",
    "a clear call to action",
    "a visually rich and dynamic scene",
    "a boring or generic corporate video",   # will be inverted
]
# last prompt is negative — we'll flip its sign
VIDEO_MEM_SIGNS = [1, 1, 1, 1, 1, -1]

BRAND_MEM_PROMPTS = [
    "a video with a clearly visible brand logo",
    "a video with strong brand identity",
    "a video where the company name is prominently displayed",
    "a generic video with no clear brand identity",             # inverted
]
BRAND_MEM_SIGNS = [1, 1, 1, -1]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_frames(video_id: str, max_frames: int = MAX_FRAMES) -> list[Image.Image]:
    """Load the first `max_frames` jpg files from frames/{video_id}/."""
    pattern = os.path.join(FRAMES_DIR, video_id, "*.jpg")
    paths = sorted(glob.glob(pattern))[:max_frames]
    return [Image.open(p).convert("RGB") for p in paths]


def batch_clip_scores(
    images: list[Image.Image],
    prompts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> np.ndarray:
    """
    Returns shape (len(images), len(prompts)) — cosine similarities.
    Processes images in batches to avoid OOM.
    """
    all_scores = []

    # Pre-encode text once — call encoder + projection explicitly to avoid
    # version-dependent return types from get_text_features()
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_out     = model.text_model(**text_inputs)
        text_features = model.text_projection(text_out.pooler_output)  # (n_prompts, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]
        img_inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_out      = model.vision_model(**img_inputs)
            img_features = model.visual_projection(img_out.pooler_output)  # (batch, D)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # (batch, n_prompts)
        scores = (img_features @ text_features.T).cpu().numpy()
        all_scores.append(scores)

    return np.vstack(all_scores)   # (n_frames, n_prompts)


def aggregate(scores: np.ndarray, signs: list[int]) -> dict:
    """
    scores : (n_frames, n_prompts)
    signs  : list of +1/-1 per prompt

    Returns scalar features: mean, max, std, slope of the signed composite score.
    """
    signed = scores * np.array(signs)          # apply negative prompt inversion
    composite = signed.mean(axis=1)            # (n_frames,) — one score per frame

    feats = {
        "mean":  float(composite.mean()),
        "max":   float(composite.max()),
        "std":   float(composite.std()),
        "slope": float(linregress(range(len(composite)), composite).slope)
                 if len(composite) > 1 else 0.0,
    }
    return feats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # Load metadata
    meta = pd.read_csv(METADATA_CSV)
    print(f"Loaded metadata: {len(meta)} videos")

    # Load CLIP
    print("Loading CLIP ViT-L/14 …")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("CLIP ready.")

    records = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Videos"):
        video_id    = row["id"]
        channel     = str(row.get("channelName", ""))

        frames = load_frames(video_id)
        if not frames:
            print(f"  [WARN] No frames found for {video_id}, skipping.")
            continue

        # ── Video memorability features ──
        v_scores = batch_clip_scores(frames, VIDEO_MEM_PROMPTS, model, processor)
        v_feats  = aggregate(v_scores, VIDEO_MEM_SIGNS)

        # ── Brand memorability features (generic) ──
        b_scores = batch_clip_scores(frames, BRAND_MEM_PROMPTS, model, processor)
        b_feats  = aggregate(b_scores, BRAND_MEM_SIGNS)

        # ── Brand memorability feature (brand-specific) ──
        brand_prompt = [f"a video clearly branded for {channel}"]
        bs_scores    = batch_clip_scores(frames, brand_prompt, model, processor)
        brand_specific_mean = float(bs_scores.mean())

        record = {"id": video_id}
        record.update({f"vmem_{k}": v for k, v in v_feats.items()})
        record.update({f"bmem_{k}": v for k, v in b_feats.items()})
        record["bmem_brand_specific"] = brand_specific_mean

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_CSV}")
    print(df.describe())


if __name__ == "__main__":
    main()