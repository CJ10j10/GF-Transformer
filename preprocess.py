#!/usr/bin/env python3
"""
Preprocess xBD dataset for GF-Transformer training.

Converts xBD format:
  images/*_pre_disaster.png, *_post_disaster.png
  labels/*_pre_disaster.json, *_post_disaster.json  (JSON with WKT polygons + damage labels)

To GF-Transformer expected format:
  images/*_pre_disaster.png, *_post_disaster.png  (unchanged)
  masks/*_pre_disaster.png   (building footprint, 255=building)
  masks/*_post_disaster.png  (damage classification: 0=bg,1=no-damage,2=minor,3=major,4=destroyed)

Usage: python preprocess.py
"""

import os, json, cv2, sys
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "xBD")
SUBDIRS = ["train", "tier3"]

DAMAGE_MAP = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0,
}

def rasterize_polygon(poly, shape):
    """Rasterize a Shapely polygon to a binary mask."""
    if poly.is_empty:
        return np.zeros(shape, dtype=np.uint8)
    if isinstance(poly, MultiPolygon):
        polys = list(poly.geoms)
    else:
        polys = [poly]

    mask = np.zeros(shape, dtype=np.uint8)
    for p in polys:
        if p.is_empty:
            continue
        coords = np.array(p.exterior.coords, dtype=np.float32)
        # Shapely WKT coordinates are (x, y), and cv2.fillPoly also expects
        # point coordinates as (x, y).
        pts = coords.astype(np.int32).reshape(1, -1, 2)
        cv2.fillPoly(mask, pts, 255)
    return mask


def process_subdir(subdir):
    """Process one data partition (train or tier3)."""
    print(f"\n{'='*60}\nProcessing: {subdir}\n{'='*60}")

    img_dir = os.path.join(DATA_DIR, subdir, "images")
    lbl_dir = os.path.join(DATA_DIR, subdir, "labels")
    tgt_dir = os.path.join(DATA_DIR, subdir, "targets")
    msk_dir = os.path.join(DATA_DIR, subdir, "masks")

    os.makedirs(msk_dir, exist_ok=True)

    # Collect pre-disaster image files
    pre_files = sorted([
        f for f in os.listdir(img_dir) if "_pre_disaster.png" in f
    ])

    print(f"  Found {len(pre_files)} pre-disaster images")

    has_targets = os.path.isdir(tgt_dir) and len(os.listdir(tgt_dir)) > 0
    print(f"  Has pre-built targets: {has_targets}")

    for fname in tqdm(pre_files, desc=f"  {subdir}"):
        base = fname.replace("_pre_disaster.png", "")
        pre_img = fname
        post_img = f"{base}_post_disaster.png"
        pre_json = f"{base}_pre_disaster.json"
        post_json = f"{base}_post_disaster.json"

        # Read image shape
        img_path = os.path.join(img_dir, pre_img)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"    WARNING: Cannot read {img_path}, skipping")
            continue
        H, W = img.shape[:2]

        # ── Step 1: Pre-disaster building footprint mask ──
        pre_target_path = os.path.join(tgt_dir, f"{base}_pre_disaster_target.png")
        if os.path.exists(pre_target_path):
            # Use pre-built target (train has it)
            msk_pre = cv2.imread(pre_target_path, cv2.IMREAD_UNCHANGED)
            if msk_pre is not None and msk_pre.max() <= 1:
                msk_pre = (msk_pre * 255).astype(np.uint8)
        else:
            # Generate from JSON (for tier3, or if target missing)
            pre_json_path = os.path.join(lbl_dir, pre_json)
            msk_pre = np.zeros((H, W), dtype=np.uint8)
            if os.path.exists(pre_json_path):
                with open(pre_json_path) as f:
                    data = json.load(f)
                xy = data["features"]["xy"]
                for bld in xy:
                    try:
                        poly = wkt.loads(bld["wkt"])
                        msk_pre = cv2.bitwise_or(msk_pre, rasterize_polygon(poly, (H, W)))
                    except Exception:
                        pass

        cv2.imwrite(os.path.join(msk_dir, pre_img), msk_pre)

        # ── Step 2: Post-disaster damage classification mask ──
        post_target_path = os.path.join(tgt_dir, f"{base}_post_disaster_target.png")
        if os.path.exists(post_target_path):
            # Use official xBD raster target when available. Values are
            # 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed.
            damage_mask = cv2.imread(post_target_path, cv2.IMREAD_UNCHANGED)
            if damage_mask is None:
                print(f"    WARNING: Cannot read {post_target_path}, falling back to JSON")
            else:
                if damage_mask.max() > 4:
                    print(f"    WARNING: Unexpected values in {post_target_path}, clipping to 0..4")
                    damage_mask = np.clip(damage_mask, 0, 4).astype(np.uint8)
                cv2.imwrite(os.path.join(msk_dir, post_img), damage_mask.astype(np.uint8))
                continue

        # Generate from JSON when no official raster target exists.
        post_json_path = os.path.join(lbl_dir, post_json)
        if not os.path.exists(post_json_path):
            # Copy pre mask as post (no damage info available)
            cv2.imwrite(os.path.join(msk_dir, post_img), msk_pre)
            continue

        damage_mask = np.zeros((H, W), dtype=np.uint8)
        with open(post_json_path) as f:
            data = json.load(f)
        xy = data["features"]["xy"]
        for bld in xy:
            subtype = bld["properties"].get("subtype", "un-classified")
            dmg_val = DAMAGE_MAP.get(subtype, 0)
            if dmg_val == 0:
                continue
            try:
                poly = wkt.loads(bld["wkt"])
                bld_mask = rasterize_polygon(poly, (H, W))
                # Assign damage value to building region
                damage_mask[bld_mask > 0] = dmg_val
                # Also set pre mask pixels for tier3 (buildings exist)
                msk_pre = cv2.bitwise_or(msk_pre, bld_mask)
            except Exception:
                pass

        # Re-save pre mask in case we added buildings from post JSON (tier3)
        cv2.imwrite(os.path.join(msk_dir, pre_img), msk_pre)
        cv2.imwrite(os.path.join(msk_dir, post_img), damage_mask)

    # Stats
    mask_files = len(os.listdir(msk_dir))
    print(f"  Generated {mask_files} mask files in masks/")


if __name__ == "__main__":
    print("=" * 60)
    print("GF-Transformer Data Preprocessing")
    print("=" * 60)

    for subdir in SUBDIRS:
        process_subdir(subdir)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
